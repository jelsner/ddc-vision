import argparse
import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# ------------------------
# Tunable params
# ------------------------
IOU_THRESH        = 0.2     # box overlap needed to match
MAX_DIST_PX       = 160     # center distance gate (pixels)
MAX_FRAME_GAP     = 2       # allow brief gaps
MIN_CONF          = 0.15    # drop very low conf
MIN_W, MIN_H      = 6, 6    # tiny-box filter

# velocity sanity (approx; in pixels/sec unless you calibrate to meters)
VEL_MAX_PX_PER_S  = 1600    # drop matches that imply faster than this
VEL_MIN_FRAMES    = 3       # need at least this many obs to consider velocity

# color locking
COLOR_FLIP_MARGIN = 0.15    # require new color’s EMA to exceed current by this margin to flip
EMA_ALPHA         = 0.3

DISC_LABELS = {"disc", "disc_red", "disc_yellow"}

def is_disc(lbl):   return (lbl or "").lower() in DISC_LABELS
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

class Track:
    def __init__(self, tid, row):
        self.id = tid
        self.rows = [row]
        self.last_frame = int(row["frame"])
        self.last_xy = (float(row["x"]), float(row["y"]))
        self.last_box = bbox_xywh_to_xyxy(row["x"], row["y"], row["width"], row["height"])
        self.family = "disc" if is_disc(row["class"]) else ("player" if is_player(row["class"]) else "other")
        # color lock for discs
        self.color_locked = False
        self.locked_color = None
        if self.family == "disc" and row["class"] in {"disc_red", "disc_yellow"}:
            self.color_locked = True
            self.locked_color = row["class"]
        # label stats
        self.class_counts = Counter([row["class"]])
        self.ema_conf = defaultdict(float)
        self.ema_conf[row["class"]] = float(row["confidence"])

    def can_consider(self, row):
        # frame gap gate
        if int(row["frame"]) - self.last_frame > MAX_FRAME_GAP:
            return False
        # family/teams gate
        fam_row = "disc" if is_disc(row["class"]) else ("player" if is_player(row["class"]) else "other")
        if self.family != fam_row:
            return False
        if self.family == "player":
            # don't mix teams
            if not same_team(self.rows[-1]["class"], row["class"]):
                return False
        return True

    def match_score(self, row, fps):
        """Return tuple (ok:boolean, score:float, vel_ok:boolean). Higher score is better."""
        # IoU + distance
        b_new = bbox_xywh_to_xyxy(row["x"], row["y"], row["width"], row["height"])
        iou = iou_xyxy(self.last_box, b_new)
        cx, cy = row["x"], row["y"]
        dx = cx - self.last_xy[0]; dy = cy - self.last_xy[1]
        dist = math.hypot(dx, dy)
        if iou < IOU_THRESH and dist > MAX_DIST_PX:
            return (False, -1.0, True)
        # velocity sanity (if enough frames seen)
        dt_frames = max(1, int(row["frame"]) - self.last_frame)
        v = (dist / dt_frames) * fps
        vel_ok = (len(self.rows) < VEL_MIN_FRAMES) or (v <= VEL_MAX_PX_PER_S)
        # score favors IoU, slightly penalizes dist
        score = iou - 0.0005*dist
        return (True, score, vel_ok)

    def add(self, row):
        lbl = row["class"]; conf = float(row["confidence"])
        # maybe update color lock (conservatively)
        if self.family == "disc" and lbl in {"disc_red", "disc_yellow"}:
            if not self.color_locked:
                self.color_locked = True
                self.locked_color = lbl
            else:
                # flipping requires EMA advantage
                ema_new = EMA_ALPHA * conf + (1-EMA_ALPHA) * self.ema_conf.get(lbl, conf)
                ema_cur = self.ema_conf.get(self.locked_color, 0.0)
                if ema_new >= ema_cur + COLOR_FLIP_MARGIN:
                    self.locked_color = lbl

        # update
        self.rows.append(row)
        self.last_frame = int(row["frame"])
        self.last_xy = (float(row["x"]), float(row["y"]))
        self.last_box = bbox_xywh_to_xyxy(row["x"], row["y"], row["width"], row["height"])
        self.class_counts[lbl] += 1
        self.ema_conf[lbl] = EMA_ALPHA * conf + (1.0 - EMA_ALPHA) * self.ema_conf.get(lbl, conf)

    def corrected_label(self):
        if self.family == "disc" and self.color_locked:
            return self.locked_color
        # otherwise: majority, tie-break by EMA
        best = None
        best_key = (-1, -1.0)
        for lbl, cnt in self.class_counts.items():
            key = (cnt, self.ema_conf.get(lbl, 0.0))
            if key > best_key:
                best_key, best = key, lbl
        return best or self.rows[-1]["class"]

def build_tracks(df, fps):
    tracks = []
    next_id = 1

    # process by frames
    for f, g in df.groupby("frame"):
        rows = g.to_dict("records")
        # candidate matches per track
        assignments = []  # (score, track_idx, row_idx, vel_ok)
        for ti, t in enumerate(tracks):
            for ri, r in enumerate(rows):
                if not t.can_consider(r): 
                    continue
                ok, score, vel_ok = t.match_score(r, fps)
                if ok:
                    assignments.append((score, ti, ri, vel_ok))
        # greedy best-first (sort by score desc)
        assignments.sort(reverse=True, key=lambda x: x[0])

        used_tracks = set()
        used_rows   = set()
        for score, ti, ri, vel_ok in assignments:
            if ti in used_tracks or ri in used_rows:
                continue
            if not vel_ok:
                continue
            tracks[ti].add(rows[ri])
            used_tracks.add(ti)
            used_rows.add(ri)

        # start new tracks for unmatched rows
        for ri, r in enumerate(rows):
            if ri in used_rows:
                continue
            t = Track(next_id, r)
            next_id += 1
            tracks.append(t)

    return tracks

def enforce_one_red_one_yellow(df):
    """
    For each frame, if there are >1 'disc_red' or >1 'disc_yellow',
    keep highest-confidence one and downgrade others to 'disc'.
    """
    for f, g in df.groupby("frame"):
        for color in ["disc_red", "disc_yellow"]:
            idx = g[g["corrected_class"] == color].index.tolist()
            if len(idx) > 1:
                # keep the one with max confidence
                keep = df.loc[idx]["confidence"].idxmax()
                for i in idx:
                    if i != keep:
                        df.at[i, "corrected_class"] = "disc"
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  default="predictions.csv")
    ap.add_argument("--out_csv", default="predictions_tracked_corrected.csv")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # basic filtering & sort
    df = df[(df["confidence"] >= MIN_CONF) &
            (df["width"] >= MIN_W) & (df["height"] >= MIN_H)].copy()
    required = ["frame","time_s","class","x","y","width","height","confidence"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df["frame"] = df["frame"].astype(int)
    df.sort_values(by=["frame","confidence"], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # build tracks
    tracks = build_tracks(df, fps=args.fps)

    # write back track_id + corrected_class
    df["track_id"] = -1
    df["corrected_class"] = df["class"]

    # Fast index by frame + (x,y,w,h,conf,class)
    # Using a map from tuple->list(indices) to update rows belonging to a track
    index = defaultdict(list)
    for i, r in df.iterrows():
        key = (r["frame"], r["x"], r["y"], r["width"], r["height"], r["confidence"], r["class"])
        index[key].append(i)

    flips = 0
    for t in tracks:
        corr = t.corrected_label()
        for r in t.rows:
            key = (r["frame"], r["x"], r["y"], r["width"], r["height"], r["confidence"], r["class"])
            for ridx in index.get(key, []):
                df.at[ridx, "track_id"] = t.id
                if df.at[ridx, "corrected_class"] != corr:
                    flips += 1
                    df.at[ridx, "corrected_class"] = corr

    # optional rule: max 1 red + 1 yellow per frame (keeps highest confidence)
    df = enforce_one_red_one_yellow(df)

    df.to_csv(args.out_csv, index=False)

    n_tracks = len(tracks)
    print(f"✅ Wrote {args.out_csv}")
    print(f"   Tracks: {n_tracks}")
    print(f"   Class corrections applied: {flips}")

if __name__ == "__main__":
    main()
