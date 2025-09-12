# render_from_csv_tracked_ms.py
import os
import csv
import cv2
import re
import numpy as np
from collections import defaultdict, deque

# === CONFIG ===
VIDEO_IN   = "/Users/jameselsner/Desktop/Escape/Rallies/ForPrediction/Rally3.mp4"
CSV_IN     = "Videos/Annotated/track_corrected_rally3.csv"   # frame-wise detections
VIDEO_OUT  = "Videos/Annotated/annotated_rally3_tracked_ms.mp4"

# Homography (pixel -> meters) saved by your click tool (np.savetxt)
H_PATH     = "H.txt"

# Optional: export an enhanced CSV with speed & metric coords
WRITE_CSV_OUT = True
CSV_OUT       = "Videos/Annotated/track_corrected_rally2_with_ms.csv"

MIN_CONF   = 0.00  # keep everything already exported; raise if needed
TRAIL_LEN  = 48    # number of past centers to draw per track
TRAIL_STEP = 1
BOX_THICK  = 3

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.4   # tuned for 4K; try 1.2–2.0
THICK      = 4
PAD_Y      = 12

# Speed smoothing / sanity
EMA_ALPHA     = 0.75          # 0..1 (higher = snappier)
SPEED_MAX_MPS = 50.0          # clamp outliers
# =============

# BGR colors
RED     = (0,   0, 255)
YELLOW  = (0, 255, 255)
BLUE    = (255, 0,   0)
WHITE   = (255, 255, 255)
GREEN   = (0, 255, 0)
BLACK   = (0, 0, 0)
GRAY    = (180, 180, 180)

def class_to_color(name: str):
    n = (name or "").lower()
    # discs
    if "disc_red" in n:     return RED, RED
    if "disc_yellow" in n:  return YELLOW, YELLOW
    if n == "disc":         return GREEN, GREEN
    # players (team A / B by substring)
    if "playera" in n or "lead_playera" in n: return BLUE, BLUE
    if "playerb" in n or "lead_playerb" in n: return WHITE, WHITE
    # default
    return GREEN, GREEN

def to_xyxy(x, y, w, h):
    x1 = int(x - w/2.0); y1 = int(y - h/2.0)
    x2 = int(x + w/2.0); y2 = int(y + h/2.0)
    return x1, y1, x2, y2

def draw_label(img, text, org, color):
    x, y = org
    # outline
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, BLACK, THICK+2, lineType=cv2.LINE_AA)
    # main
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, color, THICK, lineType=cv2.LINE_AA)

def load_by_frame(csv_path):
    """
    Returns: dict[int_frame] -> list of detections with:
      track_id (int or -1), cls (string), conf (float), x,y,w,h (floats), time_s (float or None)
    Prefers 'corrected_class' over 'class' if present.
    """
    by_frame = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        has_corr = "corrected_class" in r.fieldnames
        has_time = "time_s" in r.fieldnames
        for row in r:
            try:
                fidx = int(float(row.get("frame", 0)))
                conf = float(row.get("confidence") or 0.0)
                if conf < MIN_CONF:
                    continue
                x = float(row["x"]); y = float(row["y"])
                w = float(row["width"]); h = float(row["height"])
            except Exception:
                continue
            cls = (row.get("corrected_class") if has_corr else row.get("class")) or "obj"
            cls = str(cls)
            # track id if present
            tid = -1
            if "track_id" in row and row["track_id"] not in (None, "", "nan"):
                try: tid = int(float(row["track_id"]))
                except: tid = -1
            t_s = None
            if has_time:
                try: t_s = float(row.get("time_s"))
                except: t_s = None

            by_frame[fidx].append({
                "track_id": tid,
                "cls": cls,
                "conf": conf,
                "x": x, "y": y, "w": w, "h": h,
                "time_s": t_s
            })
    return by_frame

def px_to_meters(cx_px, cy_px, H):
    """Apply homography (pixel->meters) to a center point."""
    v = np.array([cx_px, cy_px, 1.0], dtype=np.float64)
    w = H @ v
    if abs(w[2]) < 1e-9:
        return None, None
    X = float(w[0] / w[2])
    Y = float(w[1] / w[2])
    return X, Y

def main():
    assert os.path.exists(VIDEO_IN), f"Video not found: {VIDEO_IN}"
    assert os.path.exists(CSV_IN),   f"CSV not found: {CSV_IN}"
    assert os.path.exists(H_PATH),   f"Homography not found: {H_PATH}"

    H = np.loadtxt(H_PATH)
    preds = load_by_frame(CSV_IN)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open input video")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H_px))

    # optional enhanced CSV writer
    csv_out_writer = None
    if WRITE_CSV_OUT:
        # Read input header to preserve column order + add new fields at end
        with open(CSV_IN, "r", newline="") as f_in:
            r_in = csv.DictReader(f_in)
            base_fields = r_in.fieldnames or []
        # ensure expected base fields
        add_fields = ["track_speed_ms", "center_X_m", "center_Y_m"]
        out_fields = base_fields + [f for f in add_fields if f not in base_fields]
        csv_out_writer = csv.DictWriter(open(CSV_OUT, "w", newline=""), fieldnames=out_fields)
        csv_out_writer.writeheader()

    # trail state: track_id -> deque[(x,y), ...]
    trails = {}
    # kinematic state: tid -> {last_XY_m, last_t, ema_v}
    kin = {}

    f = 0
    total_boxes = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = preds.get(f, [])
        # If time_s not provided, compute from frame/FPS
        # (We’ll still pass through provided times if present.)
        default_t = f / FPS

        # draw dets
        for d in dets:
            x1, y1, x2, y2 = to_xyxy(d["x"], d["y"], d["w"], d["h"])
            cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
            t_now = d["time_s"] if d["time_s"] is not None else default_t

            # metric center via homography
            X_m, Y_m = px_to_meters(cx, cy, H)

            # speed (EMA, clamped) for tracked objects only
            spd = None
            tid = int(d.get("track_id", -1))
            if tid != -1 and X_m is not None and Y_m is not None:
                prev = kin.get(tid)
                if prev is None:
                    kin[tid] = {"last_XY_m": (X_m, Y_m), "last_t": t_now, "ema_v": 0.0}
                    spd = 0.0
                else:
                    dt = max(1e-3, (t_now - prev["last_t"]))  # seconds
                    dist = float(np.hypot(X_m - prev["last_XY_m"][0], Y_m - prev["last_XY_m"][1]))
                    v = dist / dt
                    # EMA + clamp
                    ema_v = EMA_ALPHA * v + (1.0 - EMA_ALPHA) * prev["ema_v"]
                    ema_v = float(np.clip(ema_v, 0.0, SPEED_MAX_MPS))
                    prev.update({"last_XY_m": (X_m, Y_m), "last_t": t_now, "ema_v": ema_v})
                    spd = ema_v

            # draw box + label
            box_color, text_color = class_to_color(d["cls"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICK)

            lbl_base = d["cls"]
            if tid != -1:
                lbl_base += f" #{tid}"
            if spd is not None:
                label = f"{lbl_base} {d['conf']:.2f}  {spd:.1f} m/s"
            else:
                label = f"{lbl_base} {d['conf']:.2f}"

            draw_label(frame, label, (x1, max(0, y1 - PAD_Y)), text_color)

            # trails if we have track ids
            if tid != -1:
                if tid not in trails:
                    trails[tid] = deque(maxlen=TRAIL_LEN)
                trails[tid].append((cx, cy))

            total_boxes += 1

            # write enhanced CSV row if requested
            if csv_out_writer is not None:
                # Reconstruct a row compatible with input + added fields
                # We’ll reopen the CSV for reading raw row values? Easier: write from parsed d and fill what we know.
                # Populate minimally required fields; unknown original fields remain best-effort.
                out_row = {
                    "frame": f,
                    "time_s": t_now,
                    "class": d["cls"],
                    "confidence": d["conf"],
                    "x": d["x"], "y": d["y"],
                    "width": d["w"], "height": d["h"],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "track_id": tid if tid != -1 else ""
                }
                # add the new fields
                out_row["track_speed_ms"] = (None if spd is None else float(spd))
                out_row["center_X_m"] = (None if X_m is None else float(X_m))
                out_row["center_Y_m"] = (None if Y_m is None else float(Y_m))

                # Make sure all expected output columns exist
                for k in csv_out_writer.fieldnames:
                    if k not in out_row:
                        out_row[k] = ""
                csv_out_writer.writerow(out_row)

        # draw trails (under labels so they don’t obscure text)
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            # choose trail color by last known class if present this frame; fallback gray
            this_cls = None
            for d in dets:
                if int(d.get("track_id", -1)) == tid:
                    this_cls = d["cls"]; break
            trail_color = class_to_color(this_cls)[0] if this_cls else GRAY

            for i in range(len(pts) - 1, 0, -TRAIL_STEP):
                p1 = pts[i]; p2 = pts[i - 1]
                # fade thickness from head to tail
                t = int(np.interp(i, [1, len(pts)-1], [1, 5]))
                cv2.line(frame, p1, p2, trail_color, t)

        # HUD
        draw_label(frame, f"frame={f} det={len(dets)}", (12, 50), WHITE)

        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    if csv_out_writer:
        print(f"📄 Saved enhanced CSV with m/s: {CSV_OUT}")
    print(f"✅ Saved video: {VIDEO_OUT}  (drew {total_boxes} boxes over {f} frames)")

if __name__ == "__main__":
    main()
