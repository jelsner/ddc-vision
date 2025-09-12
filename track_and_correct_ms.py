# track_and_correct_ms.py
import math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import cv2

# ====== HARD-CODED IO ======
IN_CSV  = "Videos/Annotated/predictions_rally2.csv"   # from your RF video export
OUT_CSV = "Videos/Annotated/track_corrected_rally2.csv"
FPS     = 29.97
H_TXT   = "H.txt"   # produced by click_court_corners_6.py
# ===========================

# ------------------------
# Tunable defaults
# ------------------------
IOU_THRESH         = 0.20
MAX_DIST_PX        = 160
MAX_FRAME_GAP      = 2
MIN_CONF           = 0.15
MIN_W, MIN_H       = 6, 6

VEL_MAX_MS         = 40.0  # cap too-high speeds in m/s
VEL_MIN_FRAMES     = 3

COLOR_FLIP_MARGIN  = 0.15
EMA_ALPHA          = 0.30

DISC_SET = {"disc", "disc_red", "disc_yellow"}
def is_disc(lbl):   return (lbl or "").lower() in DISC_SET
def is_player(lbl):
    if not lbl: return False
    l = lbl.lower()
    return ("playera" in l) or ("playerb" in l) or ("lead_playera" in l) or ("lead_playerb" in l)
def same_team(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    if "playera" in a and "playera" in b: return True
    if "playerb" in a and "playerb" in b: return True
    return False

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
        if self.family == "disc" and row["class"] in {"disc_red", "disc_yellow"}:
            self.color_locked = True
            self.locked_color = row["class"]
        self.class_counts = Counter([row["class"]])
        self.ema_conf = {row["class"]: float(row["confidence"])}

    def can_consider(self, row):
        if int(row["frame"]) - self.last_frame > MAX_FRAME_GAP:
            return False
        fam_row = "disc" if is_disc(row["class"]) else ("player" if is_player(row["class"]) else "other")
        if self.family != fam_row:
            return False
        if self.family == "player":
            if not same_team(self.rows[-1]["class"], row["class"]):
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
        v_ms = dist_m / max(dt, 1e-9) if len(self.rows) >= 1 else 0.0
        row["speed_ms"] = float(v_ms)

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
    for f, g in df.groupby("frame"):
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
    for f, g in df.groupby("frame"):
        for color in ["disc_red", "disc_yellow"]:
            idx = g[g["corrected_class"] == color].index.tolist()
            if len(idx) > 1:
                keep = df.loc[idx]["confidence"].idxmax()
                for i in idx:
                    if i != keep:
                        df.at[i, "corrected_class"] = "disc"
    return df

def main():
    # Load calibration
    H = load_homography_from_txt(H_TXT)

    # Load predictions
    df = pd.read_csv(IN_CSV)
    needed = ["frame","time_s","class","x","y","width","height","confidence"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df = df[(df["confidence"] >= MIN_CONF) &
            (df["width"] >= MIN_W) & (df["height"] >= MIN_H)].copy()
    df["frame"] = df["frame"].astype(int)
    df.sort_values(by=["frame", "confidence"], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add meters columns
    X_m, Y_m = [], []
    for _, r in df.iterrows():
        Xm, Ym = px_to_m(H, float(r["x"]), float(r["y"]))
        X_m.append(Xm); Y_m.append(Ym)
    df["X_m"] = X_m; df["Y_m"] = Y_m

    # Build tracks
    tracks = build_tracks(df)

    # Write back to df
    df["track_id"] = -1
    df["corrected_class"] = df["class"]
    df["speed_ms"] = 0.0

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

    # Optional: keep at most 1 red + 1 yellow per frame
    df = enforce_one_red_one_yellow(df)

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV}")
    print(f"   Tracks: {len(tracks)}")
    print(f"   Class corrections applied: {flips}")
    print(f"   Using FPS={FPS}, VEL_MAX_MS={VEL_MAX_MS}")
    print("   Columns:", list(df.columns))

if __name__ == "__main__":
    main()
