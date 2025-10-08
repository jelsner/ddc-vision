# track_and_correct.py
import math
from collections import defaultdict, Counter, deque
import numpy as np
import pandas as pd

# ====== HARD-CODED IO ======
IN_CSV  = "Videos/Annotated/predictions_Game4.csv"   # from RF export
OUT_CSV = "Videos/Annotated/track_corrected_Game4.csv"
FPS     = 29.97
H_TXT   = "H.txt"   # 3x3 homography (px->meters)
# ===========================

# ------------------------
# Geometry (meters)
# ------------------------
COURT_SIZE   = 13.0
GAP_BETWEEN  = 17.0

def in_far_court(xm, ym, margin=0.10):
    return (-margin <= xm <= COURT_SIZE + margin) and (0.0 - margin <= ym <= COURT_SIZE + margin)

def in_near_court(xm, ym, margin=0.10):
    y0 = -(GAP_BETWEEN + COURT_SIZE)
    y1 = -GAP_BETWEEN
    return (-margin <= xm <= COURT_SIZE + margin) and (y0 - margin <= ym <= y1 + margin)

# ------------------------
# Tracker + filter params
# ------------------------
IOU_THRESH         = 0.20
MAX_DIST_PX        = 160
MAX_FRAME_GAP      = 3            # players/other
MAX_FRAME_GAP_DISC = 8            # discs can blink longer near hands

MIN_CONF           = 0.08         # allow low conf to help early frames
MIN_W, MIN_H       = 6, 6

VEL_MAX_MS         = 35.0
VEL_MIN_FRAMES     = 3

# disc-speed smoothing / sanity
SPEED_EMA_ALPHA    = 0.35  # 0..1 (higher = snappier)
ACC_MAX_UP         = 18.0  # m/s^2 max increase
ACC_MAX_DOWN       = 60.0  # m/s^2 max decrease
SPEED_MEDIAN_WIN   = 5     # rolling median (odd>=1)

EMA_ALPHA          = 0.30

# ---------- Possession proxy (keeps disc alive when occluded in a hand) ----------
PROXY_ATTACH_RADIUS_M = 0.80   # disc attaches to nearest player within this radius
PROXY_MAX_FRAMES      = 8      # max consecutive proxy frames per color
PROXY_BOX_PX          = 14     # small visual box size (pixels)
PROXY_CONF            = 0.20   # low confidence marker

# ---------- Motion-aware slot persistence tunables ----------
SWAP_GRACE        = 12     # frames: tolerate a missing slot before reassigning
EMA_VEL           = 0.50   # 0..1: EMA for per-slot velocity
MAX_CLAIM_DIST_M  = 5.0    # cap distance from predicted pos to claim a candidate

# Motion cost weights for picking replacement when a slot's TID is missing
ALPHA_DIST = 1.0   # meters term
BETA_DIR   = 0.5   # direction mismatch penalty
GAMMA_AREA = 0.15  # abrupt bbox-area jump penalty (relative)

# ---------- Classes expected from detector ----------
DISC_SET      = {"disc_red", "disc_yellow"}
PLAYER_LABELS = {"player"}   # detector class for any player

def is_disc(lbl: str) -> bool:
    return (lbl or "").lower() in DISC_SET

def is_player(lbl: str) -> bool:
    return (lbl or "").lower() in PLAYER_LABELS or "player" in (lbl or "").lower()

# ---------- Basic helpers ----------
def load_homography_from_txt(path):
    H = np.loadtxt(path).astype(np.float32)
    if H.shape != (3,3):
        raise ValueError("H.txt must contain a 3x3 matrix.")
    return H

def px_to_m(H, x, y):
    pt = np.array([x, y, 1.0], dtype=np.float32)
    w = H @ pt
    if abs(w[2]) < 1e-9:
        return (float('nan'), float('nan'))
    return float(w[0]/w[2]), float(w[1]/w[2])

def bbox_xywh_to_xyxy(x, y, w, h):
    return (x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay1-ay2) if False else (ax2-ax1)*(ay2-ay1)  # (avoid typo; keep ua positive)
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return inter / ua

# ---------- Cross-class NMS for players (dedupe double boxes) ----------
def _xyxy_from_row(r):
    return bbox_xywh_to_xyxy(r["x"], r["y"], r["width"], r["height"])

def iou_from_rows(ra, rb):
    return iou_xyxy(_xyxy_from_row(ra), _xyxy_from_row(rb))

def suppress_player_duplicates(df, iou_thresh=0.60):
    keep = []
    for f, g in df.groupby("frame", sort=True):
        idx = list(g.index)
        cand = [i for i in idx if is_player(str(df.at[i,"class"]))]
        cand = sorted(cand, key=lambda i: float(df.at[i,"confidence"]), reverse=True)
        removed = set()
        kept = []
        for i in range(len(cand)):
            if cand[i] in removed: continue
            kept.append(cand[i])
            for j in range(i+1, len(cand)):
                if cand[j] in removed: continue
                if iou_from_rows(df.loc[cand[i]], df.loc[cand[j]]) >= iou_thresh:
                    removed.add(cand[j])
        keep.extend(kept)
        keep.extend([i for i in idx if i not in cand])  # include non-player rows
    keep = sorted(set(keep))
    return df.loc[keep].reset_index(drop=True)

# ---------- Tracks ----------
class Track:
    def __init__(self, tid, row):
        self.id = tid
        self.rows = [row]
        self.last_frame = int(row["frame"])
        self.last_xy_px = (float(row["x"]), float(row["y"]))
        self.last_box_px = bbox_xywh_to_xyxy(row["x"], row["y"], row["width"], row["height"])
        self.last_xy_m  = (float(row["X_m"]), float(row["Y_m"]))
        self.family = "disc" if is_disc(row["class"]) else ("player" if is_player(row["class"]) else "other")
        self.last_speed = 0.0
        self.speed_hist = deque(maxlen=max(1, SPEED_MEDIAN_WIN))
        self.class_counts = Counter([row["class"]])
        self.ema_conf     = {row["class"]: float(row["confidence"])}

    def can_consider(self, row):
        gap = MAX_FRAME_GAP_DISC if self.family == "disc" else MAX_FRAME_GAP
        if int(row["frame"]) - self.last_frame > gap:
            return False
        fam_row = "disc" if is_disc(row["class"]) else ("player" if is_player(row["class"]) else "other")
        if self.family != fam_row:
            return False
        return True

    def match_score(self, row):
        b_new = bbox_xywh_to_xyxy(row["x"], row["y"], row["width"], row["height"])
        iou = iou_xyxy(self.last_box_px, b_new)
        cx, cy = row["x"], row["y"]
        dist_px = math.hypot(cx - self.last_xy_px[0], cy - self.last_xy_px[1])
        if iou < IOU_THRESH and dist_px > MAX_DIST_PX:
            return (False, -1.0, True)

        dt_frames = max(1, int(row["frame"]) - self.last_frame)
        dt = dt_frames / FPS
        dist_m = math.hypot(row["X_m"] - self.last_xy_m[0], row["Y_m"] - self.last_xy_m[1])
        v_ms = dist_m / max(dt, 1e-9)
        vel_ok = (len(self.rows) < VEL_MIN_FRAMES) or (v_ms <= VEL_MAX_MS)
        score = iou - 0.0005 * dist_px
        return (True, score, vel_ok)

    def add(self, row):
        lbl = row["class"]; conf = float(row["confidence"])

        dt_frames = max(1, int(row["frame"]) - self.last_frame)
        dt = dt_frames / FPS

        dist_m = math.hypot(row["X_m"] - self.last_xy_m[0], row["Y_m"] - self.last_xy_m[1])
        v_inst = dist_m / max(dt, 1e-9) if len(self.rows) >= 1 else 0.0

        if self.family == "disc":
            v_ema = SPEED_EMA_ALPHA * v_inst + (1.0 - SPEED_EMA_ALPHA) * self.last_speed
            dv = v_ema - self.last_speed
            max_up   = ACC_MAX_UP * dt
            max_down = ACC_MAX_DOWN * dt
            if dv >  max_up:     v_filt = self.last_speed + max_up
            elif dv < -max_down: v_filt = self.last_speed - max_down
            else:                v_filt = v_ema

            self.speed_hist.append(v_filt)
            v_med = (sorted(self.speed_hist)[len(self.speed_hist)//2]
                     if SPEED_MEDIAN_WIN > 1 else v_filt)

            v_final = max(0.0, min(VEL_MAX_MS, v_med))
            row["speed_ms"] = float(v_final)
            self.last_speed = v_final
        else:
            row["speed_ms"] = float(v_inst)

        self.rows.append(row)
        self.last_frame = int(row["frame"])
        self.last_xy_px = (float(row["x"]), float(row["y"]))
        self.last_box_px = bbox_xywh_to_xyxy(row["x"], row["y"], row["width"], row["height"])
        self.last_xy_m  = (float(row["X_m"]), float(row["Y_m"]))
        self.class_counts[lbl] = self.class_counts.get(lbl, 0) + 1
        self.ema_conf[lbl] = EMA_ALPHA * conf + (1.0 - EMA_ALPHA) * self.ema_conf.get(lbl, conf)

    def corrected_label(self):
        # For discs, keep raw color class if present
        if self.family == "disc":
            for lbl in ("disc_red","disc_yellow"):
                if lbl in self.class_counts:
                    return lbl
        # For players, keep majority raw label (usually "player")
        best = None
        best_key = (-1, -1.0)
        for lbl, cnt in self.class_counts.items():
            key = (cnt, self.ema_conf.get(lbl, 0.0))
            if key > best_key:
                best_key, best = key, lbl
        return best or self.rows[-1]["class"]

def build_tracks(df):
    tracks = []
    next_id = 1
    for f, g in df.groupby("frame", sort=True):
        rows = g.to_dict("records")
        cand = []
        for ti, t in enumerate(tracks):
            for ri, r in enumerate(rows):
                if not t.can_consider(r):
                    continue
                ok, score, vel_ok = t.match_score(r)
                if ok:
                    cand.append((score, ti, ri, vel_ok))
        cand.sort(reverse=True, key=lambda x: x[0])

        used_t, used_r = set(), set()
        for score, ti, ri, vel_ok in cand:
            if ti in used_t or ri in used_r:
                continue
            if not vel_ok:
                continue
            tracks[ti].add(rows[ri])
            used_t.add(ti); used_r.add(ri)

        for ri, r in enumerate(rows):
            if ri in used_r:
                continue
            t = Track(next_id, r)
            r["speed_ms"] = 0.0
            next_id += 1
            tracks.append(t)
    return tracks

# ---------- Disc selection helpers ----------
def _nearest_player_dist_m(df_frame, xm, ym):
    best = 1e9
    for _, r in df_frame.iterrows():
        if "player" in str(r["class"]).lower() or "player" in str(r.get("corrected_class", "")).lower():
            dx = float(r["X_m"]) - xm
            dy = float(r["Y_m"]) - ym
            d = math.hypot(dx, dy)
            if d < best:
                best = d
    return best

def enforce_one_red_one_yellow(df):
    """
    For each frame & color choose ONE disc using a composite score that favors:
      - detections inside either court (in-play) over outside,
      - proximity to any player (likely hand-held),
      - higher motion (speed_ms),
      - higher model confidence.
    """
    W_IN   = 2.0   # inside-court bonus
    W_NEAR = 1.2   # near-player bonus (inverse distance)
    W_SPD  = 0.7   # speed bonus
    W_CONF = 1.0   # base confidence

    for f, g in df.groupby("frame", sort=True):
        g = g.copy()
        for color in ("disc_red", "disc_yellow"):
            idxs = g.index[g["corrected_class"] == color].tolist()
            if len(idxs) <= 1:
                continue

            # Score each candidate
            scores = []
            for i in idxs:
                xm = float(df.at[i, "X_m"]); ym = float(df.at[i, "Y_m"])
                conf = float(df.at[i, "confidence"])
                spd  = float(df.at[i, "speed_ms"]) if not pd.isna(df.at[i, "speed_ms"]) else 0.0

                in_play = in_near_court(xm, ym) or in_far_court(xm, ym)
                near_d  = _nearest_player_dist_m(g, xm, ym)
                near_term = 1.0 / max(near_d, 0.4)  # saturate for very small distances

                score = (W_CONF*conf) + (W_SPD*spd) + (W_NEAR*near_term) + (W_IN*(1.0 if in_play else 0.0))
                scores.append((score, i))

            keep_i = max(scores, key=lambda t: t[0])[1]
            # Demote the others (blank them)
            for i in idxs:
                if i != keep_i:
                    df.at[i, "corrected_class"] = ""
    return df

# ---------- Possession proxy (add synthetic disc rows during hand occlusion) ----------
def add_possession_proxies(df, H):
    """
    For each frame, if a disc color is missing but a player is within PROXY_ATTACH_RADIUS_M
    of the last known disc position, add a synthetic low-conf detection at that player's center.
    """
    cols = ["frame","time_s","class","class_id","x","y","width","height",
            "confidence","x1","y1","x2","y2"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # per-frame players with pixel+meter centers
    per_frame = {}
    for f, g in df.groupby("frame", sort=True):
        items = []
        for i, r in g.iterrows():
            if is_player(str(r["class"])):
                x1,y1,x2,y2 = bbox_xywh_to_xyxy(r["x"], r["y"], r["width"], r["height"])
                cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                items.append((i, cx, cy, float(r["X_m"]), float(r["Y_m"])))
        per_frame[int(f)] = items

    # disc state
    state = {"disc_red": {"last_pos_m": None, "missing": 0},
             "disc_yellow": {"last_pos_m": None, "missing": 0}}

    new_rows = []
    frames_sorted = sorted(df["frame"].astype(int).unique())
    for f in frames_sorted:
        g = df[df["frame"] == f]
        t_sec = g["time_s"].iloc[0] if "time_s" in g.columns and len(g) else None

        for color in ("disc_red","disc_yellow"):
            present = (g["class"] == color).any()
            if present:
                gg = g[g["class"] == color].sort_values("confidence", ascending=False)
                if len(gg):
                    state[color]["last_pos_m"] = (float(gg.iloc[0]["X_m"]), float(gg.iloc[0]["Y_m"]))
                state[color]["missing"] = 0
                continue

            state[color]["missing"] += 1
            if state[color]["missing"] > PROXY_MAX_FRAMES:
                continue
            last = state[color]["last_pos_m"]
            if last is None:
                continue

            cand_players = per_frame.get(int(f), [])
            if not cand_players:
                continue
            best = None
            best_d = 1e9
            for (_idx, _cx, _cy, xm, ym) in cand_players:
                d = math.hypot(xm - last[0], ym - last[1])
                if d < best_d:
                    best_d = d; best = (_cx, _cy, xm, ym)
            if best is None or best_d > PROXY_ATTACH_RADIUS_M:
                continue

            cx, cy, xm, ym = best
            w = h = PROXY_BOX_PX
            x1 = cx - w/2.0; y1 = cy - h/2.0; x2 = cx + w/2.0; y2 = cy + h/2.0
            new_rows.append({
                "frame": int(f),
                "time_s": t_sec,
                "class": color,
                "class_id": "",
                "x": float(cx), "y": float(cy), "width": float(w), "height": float(h),
                "confidence": float(PROXY_CONF),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "X_m": float(xm), "Y_m": float(ym),
                "_is_proxy": 1
            })

    if new_rows:
        df2 = pd.DataFrame(new_rows)
        for c in df.columns:
            if c not in df2.columns:
                df2[c] = np.nan
        df = pd.concat([df, df2[df.columns]], ignore_index=True)
        df.sort_values(by=["frame","confidence"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

# ---------- Motion-aware persistent team slot assignment ----------
def assign_team_slots_persistent(df):
    """
    Motion-aware persistent team slots:
      - Seed by bbox area (2 largest -> A_1/A_2; next two smallest -> B_2/B_1 with larger= B_1).
      - Maintain per-slot motion state (pos, EMA velocity, last area).
      - If a slot's TID is missing, claim the candidate minimizing motion-aware cost.
      - No bench; extras remain "".
    """

    # persistent maps
    tid_to_slot    = {}
    slot_to_tid    = {"A_1": None, "A_2": None, "B_1": None, "B_2": None}
    slot_last_seen = {"A_1": -10**9, "A_2": -10**9, "B_1": -10**9, "B_2": -10**9}
    seeded = False

    # per-slot motion state
    slot_state = {
        "A_1": {"pos": None, "vel": np.zeros(2), "area": None},
        "A_2": {"pos": None, "vel": np.zeros(2), "area": None},
        "B_1": {"pos": None, "vel": np.zeros(2), "area": None},
        "B_2": {"pos": None, "vel": np.zeros(2), "area": None},
    }

    def is_player_row(i):
        lbl = str(df.at[i, "class"]).lower()
        cc  = str(df.at[i, "corrected_class"]).lower() if "corrected_class" in df.columns else lbl
        return ("player" in lbl) or ("player" in cc)

    def label_slot(i, slot):
        team = "playerA" if slot.startswith("A_") else "playerB"
        num  = "1" if slot.endswith("_1") else "2"
        df.at[i, "corrected_class"] = f"{team}_{num}"

    def update_slot_state(slot, xm, ym, area, dt):
        st = slot_state[slot]
        p_now = np.array([xm, ym], dtype=float)
        if st["pos"] is None:
            st["vel"] = np.zeros(2)
        else:
            v_inst = (p_now - st["pos"]) / max(dt, 1e-3)
            st["vel"] = EMA_VEL * v_inst + (1.0 - EMA_VEL) * st["vel"]
        st["pos"]  = p_now
        st["area"] = area

    # precompute frame->time
    frame_time = {}
    for f, g in df.groupby("frame", sort=True):
        if "time_s" in g.columns and g["time_s"].notna().any():
            frame_time[int(f)] = float(g["time_s"].iloc[0])
        else:
            frame_time[int(f)] = int(f) / FPS

    prev_t = None

    for f, g in df.groupby("frame", sort=True):
        t_now = frame_time[int(f)]
        dt = (t_now - prev_t) if (prev_t is not None) else (1.0 / FPS)
        prev_t = t_now

        idxs = g.index.tolist()

        # Clear player labels for this frame
        for i in idxs:
            if is_player_row(i):
                df.at[i, "corrected_class"] = ""

        # Candidate players
        cand = []
        for i in idxs:
            if not is_player_row(i):
                continue
            tid_raw = df.at[i, "track_id"]
            if pd.isna(tid_raw):
                continue
            tid = int(tid_raw)
            if tid < 0:
                continue
            xm = float(df.at[i, "X_m"]); ym = float(df.at[i, "Y_m"])
            w  = float(df.at[i, "width"]); h = float(df.at[i, "height"])
            area = max(1.0, w*h)
            cand.append({"i": i, "tid": tid, "xm": xm, "ym": ym, "area": area})

        if not cand:
            continue

        # ---------- seeding by area (first workable frame) ----------
        if not seeded:
            if len(cand) >= 4:
                by_area_desc = sorted(cand, key=lambda c: c["area"], reverse=True)
                a1, a2 = by_area_desc[0], by_area_desc[1]
                by_area_asc = sorted(by_area_desc[2:], key=lambda c: c["area"])
                b2, b1 = by_area_asc[0], by_area_asc[1]   # larger of the two small = B_1

                slot_to_tid["A_1"] = a1["tid"]; slot_last_seen["A_1"] = f
                slot_to_tid["A_2"] = a2["tid"]; slot_last_seen["A_2"] = f
                slot_to_tid["B_1"] = b1["tid"]; slot_last_seen["B_1"] = f
                slot_to_tid["B_2"] = b2["tid"]; slot_last_seen["B_2"] = f
                for s, tslot in slot_to_tid.items():
                    if tslot is not None:
                        tid_to_slot[tslot] = s

                for slot, c in [("A_1", a1), ("A_2", a2), ("B_1", b1), ("B_2", b2)]:
                    update_slot_state(slot, c["xm"], c["ym"], c["area"], dt)

                present = {c["tid"]: c for c in cand}
                for s in ["A_1","A_2","B_1","B_2"]:
                    tslot = slot_to_tid[s]
                    if tslot in present:
                        label_slot(present[tslot]["i"], s)

                seeded = True
            else:
                continue  # wait until 4 players to seed

        # ---------- persistence with motion ----------
        assigned_tids = set()
        present_by_tid = {c["tid"]: c for c in cand}

        # 1) keep existing TIDs if visible
        for s in ["A_1","A_2","B_1","B_2"]:
            tid = slot_to_tid[s]
            if tid is not None and tid in present_by_tid:
                c = present_by_tid[tid]
                label_slot(c["i"], s)
                slot_last_seen[s] = f
                update_slot_state(s, c["xm"], c["ym"], c["area"], dt)
                assigned_tids.add(tid)

        # 2) claim best candidate for any missing slot using motion-aware cost
        def motion_cost(slot, c):
            st = slot_state[slot]
            if st["pos"] is None:
                # no prior; weaker constraint
                d = 0.0
                dir_pen = 0.0
            else:
                p_pred = st["pos"] + st["vel"] * max(dt, 1e-3)
                d = float(np.linalg.norm(np.array([c["xm"], c["ym"]]) - p_pred))
                v = st["vel"]
                if np.linalg.norm(v) < 1e-6:
                    dir_pen = 0.0
                else:
                    v_unit = v / (np.linalg.norm(v) + 1e-9)
                    step   = np.array([c["xm"], c["ym"]]) - st["pos"]
                    if np.linalg.norm(step) < 1e-6:
                        dir_pen = 0.0
                    else:
                        step_unit = step / (np.linalg.norm(step) + 1e-9)
                        cos_sim = float(np.clip(np.dot(v_unit, step_unit), -1.0, 1.0))
                        dir_pen = (1.0 - cos_sim)
            if slot_state[slot]["area"] is None:
                area_pen = 0.0
            else:
                a0 = slot_state[slot]["area"]; a1 = c["area"]
                area_pen = abs(a1 - a0) / max(a0, a1, 1.0)
            return ALPHA_DIST * d + BETA_DIR * dir_pen + GAMMA_AREA * area_pen, d

        for s in ["A_1","A_2","B_1","B_2"]:
            tid = slot_to_tid[s]
            if tid is not None and tid in present_by_tid:
                continue  # already assigned this frame

            recently_seen = (tid is not None) and ((f - slot_last_seen[s]) < SWAP_GRACE)
            pool = [c for c in cand if c["tid"] not in assigned_tids]
            if not pool:
                continue

            scored = []
            for c in pool:
                # don't steal a TID already bound to a different slot
                if (c["tid"] in tid_to_slot) and (tid_to_slot[c["tid"]] not in (None, s)):
                    continue
                cost, dist_pred = motion_cost(s, c)
                scored.append((cost, dist_pred, c))
            if not scored:
                continue

            scored.sort(key=lambda t: t[0])
            best_cost, best_dist, best = scored[0]

            if recently_seen and best_dist > MAX_CLAIM_DIST_M:
                # too far from prediction; keep waiting
                continue

            # claim
            slot_to_tid[s] = best["tid"]
            tid_to_slot[best["tid"]] = s
            slot_last_seen[s] = f
            label_slot(best["i"], s)
            update_slot_state(s, best["xm"], best["ym"], best["area"], dt)
            assigned_tids.add(best["tid"])

        # 3) everyone else remains "" for this frame

    return df

# =================== MAIN ===================
def main():
    H = load_homography_from_txt(H_TXT)

    df = pd.read_csv(IN_CSV)
    needed = ["frame","time_s","class","x","y","width","height","confidence"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # If your model sometimes outputs "playerA_1", etc., normalize to "player"
    # df["class"] = df["class"].astype(str)
    # df.loc[df["class"].str.contains(r"^player", case=False, regex=True), "class"] = "player"
    # df.loc[df["class"].str.contains(r"lead_player", case=False, regex=True), "class"] = "player"
    # Optional: coerce 'obj' people to 'player'
    # df.loc[df["class"].str.fullmatch(r"obj", case=False), "class"] = "player"

    # basic filtering
    df = df[(df["confidence"] >= MIN_CONF) &
            (df["width"] >= MIN_W) & (df["height"] >= MIN_H)].copy()
    df["frame"] = df["frame"].astype(int)
    df.sort_values(by=["frame","confidence"], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # de-dup player double boxes
    df = suppress_player_duplicates(df, iou_thresh=0.60)

    # add meter coords
    Xs, Ys = [], []
    for _, r in df.iterrows():
        Xm, Ym = px_to_m(H, float(r["x"]), float(r["y"]))
        Xs.append(Xm); Ys.append(Ym)
    df["X_m"] = Xs; df["Y_m"] = Ys

    # Possession proxies for missing discs near hands
    df = add_possession_proxies(df, H)

    # Build tracks
    tracks = build_tracks(df)

    # write back basic track info & default corrected_class
    df["track_id"]        = -1
    df["corrected_class"] = df["class"]
    df["speed_ms"]        = 0.0

    # map rows -> track updates
    idx_map = defaultdict(list)
    for i, r in df.iterrows():
        key = (r["frame"], r["x"], r["y"], r["width"], r["height"], r["confidence"], r["class"])
        idx_map[key].append(i)

    flips = 0
    for t in tracks:
        corr = t.corrected_label()
        for r in t.rows:
            key = (r["frame"], r["x"], r["y"], r["width"], r["height"], r["confidence"], r["class"])
            for ridx in idx_map.get(key, []):
                df.at[ridx, "track_id"] = t.id
                df.at[ridx, "speed_ms"] = float(r.get("speed_ms", 0.0))
                # Keep raw for players; discs already set by corr
                if is_disc(df.at[ridx, "class"]):
                    if df.at[ridx, "corrected_class"] != corr:
                        flips += 1
                        df.at[ridx, "corrected_class"] = corr
                else:
                    df.at[ridx, "corrected_class"] = df.at[ridx, "class"]

    # Start with discs set, then enforce at most one per color per frame
    df.loc[df["class"].isin(["disc_red","disc_yellow"]), "corrected_class"] = df["class"]
    df = enforce_one_red_one_yellow(df)

    # Team A/B + 1/2 assignment (persistent, motion-aware)
    df = assign_team_slots_persistent(df)

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV}")
    print(f"   Tracks: {len(tracks)}")
    print(f"   Class corrections adjusted: {flips}")
    print(f"   Using FPS={FPS}, VEL_MAX_MS={VEL_MAX_MS}")
    print("   Columns:", list(df.columns))

if __name__ == "__main__":
    main()
