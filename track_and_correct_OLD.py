# track_and_correct.py
import math
from collections import defaultdict, Counter, deque
import numpy as np
import pandas as pd

# ====== HARD-CODED IO ======
IN_CSV  = "Videos/Annotated/predictions_G2_V1-V10.csv"   # from RF export
OUT_CSV = "Videos/Annotated/track_corrected_G2_V1-V10.csv"
FPS     = 29.97
H_TXT   = "H.txt"   # 3x3 homography (px->meters) from click_court_corners_8.py
# ===========================

# ------------------------
# Geometry (meters)
# ------------------------
COURT_SIZE   = 13.0
GAP_BETWEEN  = 17.0

# Far court world Y in [0, 13]
# Near court world Y in [-(GAP+13), -GAP]
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

COLOR_FLIP_MARGIN  = 0.15
EMA_ALPHA          = 0.30

# Possession proxy (keeps disc alive when occluded in a hand)
PROXY_ATTACH_RADIUS_M = 0.80   # disc attaches to nearest player within this radius
PROXY_MAX_FRAMES      = 8      # max consecutive proxy frames per color
PROXY_BOX_PX          = 14     # small visual box size (pixels)
PROXY_CONF            = 0.20   # low confidence marker

# Classes
DISC_SET        = {"disc_red", "disc_yellow"}  # no generic "disc"
PLAYER_TOKENS   = ("playera_", "playerb_", "lead_playera", "lead_playerb", "player")  # permissive

def is_disc(lbl: str) -> bool:
    return (lbl or "").lower() in DISC_SET

def is_player(lbl: str) -> bool:
    if not lbl: return False
    l = lbl.lower()
    return any(tok in l for tok in PLAYER_TOKENS)

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
        cand = [i for i in idx if is_player(str(df.at[i,"class"])) or "player" in str(df.at[i,"class"]).lower()]
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
        # keep all non-player rows too
        keep.extend([i for i in idx if i not in cand])
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
        self.color_locked = False
        self.locked_color = None
        self.last_speed = 0.0
        self.speed_hist = deque(maxlen=max(1, SPEED_MEDIAN_WIN))

        if self.family == "disc" and row["class"] in {"disc_red", "disc_yellow"}:
            self.color_locked = True
            self.locked_color = row["class"]

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

        if self.family == "disc" and lbl in {"disc_red", "disc_yellow"}:
            if not self.color_locked:
                self.color_locked = True
                self.locked_color = lbl
            else:
                ema_new = EMA_ALPHA * conf + (1-EMA_ALPHA) * self.ema_conf.get(lbl, conf)
                ema_cur = self.ema_conf.get(self.locked_color, 0.0)
                if ema_new >= ema_cur + COLOR_FLIP_MARGIN:
                    self.locked_color = lbl

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
        if self.family == "disc" and self.color_locked:
            return self.locked_color
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

def enforce_one_red_one_yellow(df):
    for f, g in df.groupby("frame", sort=True):
        for color in ["disc_red", "disc_yellow"]:
            idx = g[g["corrected_class"] == color].index.tolist()
            if len(idx) > 1:
                keep = df.loc[idx]["confidence"].idxmax()
                for i in idx:
                    if i != keep:
                        # demote extras to the same color but mark as secondary? simplest: drop to lowest confidence
                        df.at[i, "corrected_class"] = color
                        # Optionally you could blank these or set a flag; we leave as-is.
    return df

# ---------- Possession proxy (add synthetic disc rows during hand occlusion) ----------
def add_possession_proxies(df, H):
    """
    For each frame, if a disc color is missing but a player is within PROXY_ATTACH_RADIUS_M
    of the last known disc position, add a synthetic low-conf detection at that player's center.
    """
    cols = ["frame","time_s","class","class_id","x","y","width","height",
            "confidence","x1","y1","x2","y2"]
    ensure_cols = [c for c in cols if c not in df.columns]
    for c in ensure_cols:
        df[c] = np.nan

    # build quick access: per-frame players with pixel+meter centers
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
    state = {
        "disc_red":    {"last_pos_m": None, "missing": 0},
        "disc_yellow": {"last_pos_m": None, "missing": 0},
    }

    new_rows = []
    frames_sorted = sorted(df["frame"].astype(int).unique())
    for f in frames_sorted:
        g = df[df["frame"] == f]
        t_sec = g["time_s"].iloc[0] if "time_s" in g.columns and len(g) else None

        for color in ("disc_red","disc_yellow"):
            # present?
            present = (g["class"] == color).any()
            if present:
                # update last_pos_m from highest-conf one
                gg = g[g["class"] == color].sort_values("confidence", ascending=False)
                if len(gg):
                    state[color]["last_pos_m"] = (float(gg.iloc[0]["X_m"]), float(gg.iloc[0]["Y_m"]))
                state[color]["missing"] = 0
                continue

            # missing
            state[color]["missing"] += 1
            if state[color]["missing"] > PROXY_MAX_FRAMES:
                continue
            last = state[color]["last_pos_m"]
            if last is None:
                continue

            # find nearest player in this frame
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
            # create synthetic small box around player center
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
        # ensure same dtypes
        for c in df.columns:
            if c not in df2.columns:
                df2[c] = np.nan
        df = pd.concat([df, df2[df.columns]], ignore_index=True)
        df.sort_values(by=["frame","confidence"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

# ---------- Persistent team slot assignment ----------
def assign_team_slots_persistent(df):
    """
    After tracks exist, assign playerA_1/A_2 (near court) and playerB_1/B_2 (far court)
    with sticky slots keyed by track_id to avoid identity flip.
    """
    # slot memory per court: track_id -> slot
    near_tid_to_slot = {}   # tid -> "1"/"2"
    far_tid_to_slot  = {}   # tid -> "1"/"2"
    # last known slot positions to help re-acquire after brief misses
    near_lastpos = {"1": None, "2": None}
    far_lastpos  = {"1": None, "2": None}

    # Working column
    df["corrected_class"] = df["corrected_class"].astype(str)

    for f, g in df.groupby("frame", sort=True):
        # collect players in each court (use corrected_class if already set, else fallback)
        cand = []
        for i in g.index:
            if not is_player(str(df.at[i,"corrected_class"])) and not is_player(str(df.at[i,"class"])):
                continue
            xm, ym = float(df.at[i,"X_m"]), float(df.at[i,"Y_m"])
            tid = int(df.at[i,"track_id"]) if "track_id" in df.columns and not pd.isna(df.at[i,"track_id"]) else -1
            conf = float(df.at[i,"confidence"])
            if in_near_court(xm, ym):
                cand.append(("near", i, tid, xm, ym, conf))
            elif in_far_court(xm, ym):
                cand.append(("far",  i, tid, xm, ym, conf))

        # split and keep top-2 by confidence per court
        near = [(i, tid, xm, ym, conf) for court,i,tid,xm,ym,conf in cand if court=="near"]
        far  = [(i, tid, xm, ym, conf) for court,i,tid,xm,ym,conf in cand if court=="far"]

        near = sorted(near, key=lambda t: (-t[4], t[2]))[:2]
        far  = sorted(far,  key=lambda t: (-t[4], t[2]))[:2]

        # helper: assign slots with stickiness
        def assign_side(rows, tid2slot, lastpos):
            out = []  # (row_index, label_text)
            # order by X (left->right) for a deterministic bias
            rows = sorted(rows, key=lambda t: t[2])  # using tid as tiebreaker already sorted by conf; we’ll re-order by X below
            rows = sorted(rows, key=lambda t: t[2])  # dummy; we’ll compute X from df (next)
            rows = sorted(rows, key=lambda t: float(df.at[t[0],"X_m"]))  # left->right

            # Which slots are free?
            slots = {"1": None, "2": None}
            # First give back previously owned slots
            for (i, tid, xm, ym, conf) in rows:
                if tid in tid2slot:
                    s = tid2slot[tid]
                    slots[s] = tid

            # fill remaining by proximity to lastpos (sticky)
            unknown = [(i, tid, float(df.at[i,"X_m"]), float(df.at[i,"Y_m"])) for (i,tid,_,_,_) in rows if tid not in tid2slot]
            open_slots = [s for s in ("1","2") if slots[s] is None]

            def cost(xm, ym, slot):
                lp = lastpos[slot]
                if lp is None:
                    # slight left/right bias (slot 1 = left)
                    return (xm + (0.25 if slot=="1" else -0.25))**2 + ym*ym
                return (xm - lp[0])**2 + (ym - lp[1])**2

            for (i, tid, xm, ym) in sorted(unknown, key=lambda z: min(cost(z[2], z[3], s) for s in open_slots) if open_slots else 0):
                if not open_slots:
                    break
                best_slot = min(open_slots, key=lambda s: cost(xm, ym, s))
                lp = lastpos[best_slot]
                if lp is not None:
                    # avoid huge teleport flips
                    if (xm - lp[0])**2 + (ym - lp[1])**2 > (3.0**2):
                        continue
                tid2slot[tid] = best_slot
                slots[best_slot] = tid
                lastpos[best_slot] = (xm, ym)
                open_slots.remove(best_slot)

            # emit labels
            for s in ("1","2"):
                if slots[s] is None: continue
                # find row index for this tid
                for (i, tid, xm, ym, conf) in rows:
                    if tid == slots[s]:
                        out.append((i, s))
                        lastpos[s] = (xm, ym)
                        break
            return out

        near_asg = assign_side(near, near_tid_to_slot, near_lastpos)
        far_asg  = assign_side(far,  far_tid_to_slot,  far_lastpos)

        # set labels
        for i, s in near_asg:
            df.at[i, "corrected_class"] = f"playerA_{s}"
        for i, s in far_asg:
            df.at[i, "corrected_class"] = f"playerB_{s}"

        # any remaining in-court players not assigned -> bench (rare)
        for court,i,tid,xm,ym,conf in cand:
            if "playerA_" in str(df.at[i,"corrected_class"]) or "playerB_" in str(df.at[i,"corrected_class"]):
                continue
            df.at[i, "corrected_class"] = "player_bench"

    return df

# =================== MAIN ===================
def main():
    H = load_homography_from_txt(H_TXT)

    df = pd.read_csv(IN_CSV)
    needed = ["frame","time_s","class","x","y","width","height","confidence"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

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

    # possession proxies for missing discs near hands
    df = add_possession_proxies(df, H)

    # Build tracks
    tracks = build_tracks(df)

    # write back basic track info
    df["track_id"]        = -1
    df["corrected_class"] = df["class"]
    df["speed_ms"]        = 0.0

    # fast index to map rows belonging to a track
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
                if df.at[ridx, "corrected_class"] != corr:
                    flips += 1
                    df.at[ridx, "corrected_class"] = corr

    # Only one red + one yellow per frame (keep highest conf)
    df = enforce_one_red_one_yellow(df)

    # Persistent A/B 1/2 per court (sticky)
    df = assign_team_slots_persistent(df)

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV}")
    print(f"   Tracks: {len(tracks)}")
    print(f"   Class corrections applied: {flips}")
    print(f"   Using FPS={FPS}, VEL_MAX_MS={VEL_MAX_MS}")
    print("   Columns:", list(df.columns))

if __name__ == "__main__":
    main()
