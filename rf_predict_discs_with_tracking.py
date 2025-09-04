# ddc_predict_discs_with_tracking_trails_vclass.py
import os
import cv2
import csv
import base64
import requests
import numpy as np
from collections import deque

# ====== CONFIG ======
API_KEY   = "XXXX"
MODEL_ID  = "ddc-discs-5vdna/8"    # RF-DETR v8
VIDEO_IN  = "/Users/jameselsner/Desktop/Projects/ddc-vision/Videos/Jim_Vergil.mov"  # full path
VIDEO_OUT = "predicted_tracked6.mp4"
CSV_OUT   = "tracks6.csv"

CONFIDENCE   = 0.21
FRAME_STRIDE = 1
RESIZE_WIDTH = 1536
TOP_K        = 2

# === Court calibration (homography) ===
# Image points (px): top-left, top-right, bottom-right, bottom-left of ONE 13m x 13m court
IMG_PTS = np.array([
    [468, 844],
    [1371, 853],
    [1501, 889],
    [348, 870]
], dtype=np.float32)

# World points (meters) for the same four corners; make the top-left court start at (0,0)
# Example: one 13x13 square
WRLD_PTS = np.array([
    [0.0,  0.0],
    [13.0, 0.0],
    [13.0, 13.0],
    [0.0,  13.0]
], dtype=np.float32)

H, _ = cv2.findHomography(IMG_PTS, WRLD_PTS, method=cv2.RANSAC)

def img_to_meters(x, y):
    """Project image (px) center to court meters via homography."""
    pt = np.array([[x, y, 1.0]], dtype=np.float32).T  # 3x1
    wpt = (H @ pt).ravel()
    return float(wpt[0] / wpt[2]), float(wpt[1] / wpt[2])  # (X_m, Y_m)

# --- Speed sanity + signed direction config ---
MAX_SPEED_MS = 35.0   # cap physically-plausible disc speed
REVERSE_DIR  = False  # flip sign if your camera orientation is opposite

# Build a unit "downrange" direction in WORLD (meters) using the court geometry:
# vector from bottom edge midpoint -> top edge midpoint (usually toward far court at top of image)
top_mid_img = (IMG_PTS[0] + IMG_PTS[1]) * 0.5
bot_mid_img = (IMG_PTS[3] + IMG_PTS[2]) * 0.5

top_m = img_to_meters(float(top_mid_img[0]), float(top_mid_img[1]))
bot_m = img_to_meters(float(bot_mid_img[0]), float(bot_mid_img[1]))

u = np.array([top_m[0] - bot_m[0], top_m[1] - bot_m[1]], dtype=np.float32)
if REVERSE_DIR:
    u = -u
U_HAT = u / (np.linalg.norm(u) + 1e-9)  # unit vector in meters, near -> far

# Trail drawing
TRAIL_LEN       = 32   # recent points kept per track
TRAIL_MIN_STEP  = 2    # draw every Nth segment
TRAIL_THICK_MAX = 5    # line thickness near head
TRAIL_THICK_MIN = 1    # line thickness near tail
TRAIL_COLOR     = (0, 255, 255)  # BGR (yellow)
TRAIL_AGE_DROP  = 90   # drop trail if not seen for N frames

# Kinematics state
kin_state = {}  # tid -> {last_xy_m, last_frame, ema_v (m/s)}
ALPHA_V = 0.3   # EMA smoothing for speed

# Optional ByteTrack (via supervision). Falls back to a tiny IoU tracker.
USE_BYTETRACK = True
try:
    from supervision import Detections
    from supervision.draw.color import ColorPalette
    from supervision.annotators.core import BoxAnnotator, LabelAnnotator
    from supervision.tracker.byte_tracker import BYTETracker, BYTETrackerArgs
    HAVE_SUPERVISION = True
except Exception:
    HAVE_SUPERVISION = False
    USE_BYTETRACK = False

BT_FRAME_RATE   = 30 if FRAME_STRIDE == 1 else 15
BT_TRACK_THRESH = CONFIDENCE
BT_MATCH_THRESH = 0.88
BT_TRACK_BUFFER = 60

class NaiveTracker:
    def __init__(self, iou_thresh=0.35, max_age=45):
        self.next_id = 1
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    @staticmethod
    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        bb = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + bb - inter + 1e-6)

    def update(self, boxes):
        # age and drop stale
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        assigned = {}
        # greedy IoU assignment
        for b in boxes:
            best_tid, best_iou = None, 0.0
            for tid, info in self.tracks.items():
                iou = self.iou(b, info["bbox"])
                if iou > best_iou:
                    best_tid, best_iou = tid, iou
            if best_iou >= self.iou_thresh:
                self.tracks[best_tid] = {"bbox": b, "age": 0}
                assigned[tuple(b)] = best_tid
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"bbox": b, "age": 0}
                assigned[tuple(b)] = tid

        return [(assigned[tuple(b)], b) for b in boxes]

_last_scale = 1.0

def rf_infer(frame_bgr):
    """Roboflow Hosted Inference → JSON."""
    global _last_scale
    if RESIZE_WIDTH is not None:
        h, w = frame_bgr.shape[:2]
        _last_scale = RESIZE_WIDTH / float(w)
        frame_bgr = cv2.resize(frame_bgr, (RESIZE_WIDTH, int(h * _last_scale)))
    else:
        _last_scale = 1.0

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return {"predictions": []}

    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    r = requests.post(
        f"https://detect.roboflow.com/{MODEL_ID}",
        params={"api_key": API_KEY, "confidence": CONFIDENCE},
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    return r.json() if r.ok else {"predictions": []}

def to_xyxy_conf_cls(preds, classes_map=None):
    """
    Robustly extract: boxes, confidence, class name, class_id.
    Handles 'class'/'label' as string, or numeric ids with a classes map.
    """
    xyxy, conf, cls, class_ids = [], [], [], []
    inv_map = None
    if isinstance(classes_map, dict):
        # {"0":"disc_red","1":"disc_yellow","2":"disc"} -> {"disc_red":0,...}
        inv_map = {v: int(k) for k, v in classes_map.items()}

    for d in preds:
        x, y, w, h = d["x"], d["y"], d["width"], d["height"]
        xyxy.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        conf.append(d.get("confidence", 0.0))

        cname = (d.get("class") or d.get("label") or "disc").lower()
        cls.append(cname)

        if "class_id" in d and isinstance(d["class_id"], (int, float)):
            cid = int(d["class_id"])
        elif isinstance(d.get("class"), (int, float)):
            cid = int(d["class"])
        elif inv_map is not None and cname in inv_map:
            cid = inv_map[cname]
        else:
            cid = 0
        class_ids.append(cid)

    if not xyxy:
        return (np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                [], np.empty((0,), np.int32))

    return (np.array(xyxy, np.float32),
            np.array(conf, np.float32),
            cls,
            np.array(class_ids, dtype=np.int32))

def draw_trails(frame, trails):
    """Draw fading polylines for each track's recent centers."""
    for tid, pts in trails.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1, 0, -TRAIL_MIN_STEP):
            p1 = pts[i]
            p2 = pts[i - 1]
            t = int(np.interp(i, [1, len(pts) - 1], [TRAIL_THICK_MIN, TRAIL_THICK_MAX]))
            cv2.line(frame, p1, p2, TRAIL_COLOR, thickness=t)
        cv2.circle(frame, pts[-1], 3, (0, 255, 0), -1)

def main():
    assert os.path.exists(VIDEO_IN), f"Video not found: {VIDEO_IN}"
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open the video.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    # Tracker
    if USE_BYTETRACK and HAVE_SUPERVISION:
        tracker = BYTETracker(BYTETrackerArgs(
            track_thresh=BT_TRACK_THRESH,
            match_thresh=BT_MATCH_THRESH,
            track_buffer=BT_TRACK_BUFFER,
            frame_rate=BT_FRAME_RATE
        ))
        palette = ColorPalette.default()
        box_annotator   = BoxAnnotator(color=palette, thickness=2)
        label_annotator = LabelAnnotator(color=palette)
        use_bt = True
        print("Tracker: ByteTrack")
    else:
        tracker = NaiveTracker(iou_thresh=0.35, max_age=45)
        use_bt = False
        print("Tracker: Naive IoU fallback")

    # Trails state
    trails = {}           # track_id -> deque([(x,y), ...])
    last_seen = {}        # track_id -> frames since seen

    total, with_det = 0, 0

    with open(CSV_OUT, "w", newline="") as fcsv:
        log = csv.writer(fcsv)
        log.writerow([
            "frame","track_id","class","x1","y1","x2","y2","conf",
            "cx_px","cy_px","X_m","Y_m","speed_ms","vel_signed_ms"
        ])

        f = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if f % FRAME_STRIDE != 0:
                writer.write(frame)
                f += 1
                continue

            # ==== INFERENCE ====
            data = rf_infer(frame)
            preds = data.get("predictions", [])
            classes_map = data.get("classes")  # may be None

            xyxy, conf, cls, class_ids = to_xyxy_conf_cls(preds, classes_map=classes_map)
            if _last_scale != 1.0 and len(xyxy):
                xyxy /= _last_scale

            # ==== FILTERING (apply BEFORE counting and TOP_K) ====
            def min_conf_for_class(cl, base=CONFIDENCE):
                name = (cl or "disc").lower()
                if "yellow" in name:
                    return max(0.18, base - 0.02)
                if "red" in name:
                    return max(0.20, base)
                return base

            filtered_boxes, filtered_conf, filtered_cls, filtered_ids = [], [], [], []
            for b, c, cl, cid in zip(xyxy, conf, cls, class_ids):
                x1, y1, x2, y2 = map(int, b)
                w, h = x2 - x1, y2 - y1
                ar = max(w, h) / float(min(w, h) + 1e-6)

                # Shape gating
                if ar > 2.0:
                    continue

                # Confidence gating (per class)
                if c < min_conf_for_class(cl):
                    continue

                filtered_boxes.append(b)
                filtered_conf.append(c)
                filtered_cls.append(cl)
                filtered_ids.append(cid)

            xyxy = np.array(filtered_boxes, np.float32) if filtered_boxes else np.empty((0, 4), np.float32)
            conf = np.array(filtered_conf,  np.float32) if filtered_conf else np.empty((0,), np.float32)
            cls  = filtered_cls
            class_ids = np.array(filtered_ids, np.int32) if filtered_ids else np.empty((0,), np.int32)

            # Keep at most TOP_K after filtering
            if len(conf) > TOP_K:
                idx = np.argsort(-conf)[:TOP_K]
                xyxy, conf = xyxy[idx], conf[idx]
                cls       = [cls[i] for i in idx]
                class_ids = class_ids[idx] if len(class_ids) else np.zeros(len(idx), np.int32)

            # Count AFTER filtering so stats match what we draw
            total += 1
            if len(xyxy) > 0:
                with_det += 1

            # ==== TRACKING & DRAWING ====
            if use_bt:
                det = Detections(xyxy=xyxy, confidence=conf, class_id=class_ids) if len(xyxy) else Detections.empty()
                tracked = tracker.update_with_detections(det)
                seen_now = set()

                if len(tracked) > 0:
                    # Per-track update: meters, speed, CSV, trails
                    for i, box in enumerate(tracked.xyxy):
                        tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                        conf_i = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                        cid = int(tracked.class_id[i]) if tracked.class_id is not None else 0

                        # Map class id → name for CSV
                        if isinstance(classes_map, dict) and str(cid) in classes_map:
                            cname = classes_map[str(cid)]
                        else:
                            cname = cls[i] if i < len(cls) else "disc"
                        
                        # Box & center
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Meters (current)
                        X_m, Y_m = img_to_meters(cx, cy)

                        # --- Update speed BEFORE writing the row ---
                        now = f
                        prev = kin_state.get(tid)
                        if prev is None:
                            ema_v = 0.0
                            ema_signed = 0.0
                            kin_state[tid] = {"last_xy_m": (X_m, Y_m), "last_frame": now,
                                              "ema_v": ema_v, "ema_signed": ema_signed}
                        else:
                            px, py = prev["last_xy_m"]
                            dt = max(1, now - prev["last_frame"]) / FPS  # seconds

                        # Displacement in meters
                            dx = X_m - px
                            dy = Y_m - py
                            step = float(np.hypot(dx, dy))

                            # --- Clamp per-frame step to MAX_SPEED_MS to kill spikes ---
                            max_step = MAX_SPEED_MS * dt
                            if step > max_step and step > 0:
                                scale = max_step / step
                                dx *= scale
                                dy *= scale
                                X_m = px + dx
                                Y_m = py + dy
                                step = max_step  # after clamp

                            raw_speed = step / dt                 # m/s (non-negative)
                            signed_v  = float((dx * U_HAT[0] + dy * U_HAT[1]) / dt)  # m/s (signed)

                            # EMA smoothing
                            ema_v      = ALPHA_V * raw_speed + (1.0 - ALPHA_V) * prev["ema_v"]
                            ema_signed = ALPHA_V * signed_v  + (1.0 - ALPHA_V) * prev["ema_signed"]

                            prev.update({"last_xy_m": (X_m, Y_m), "last_frame": now,
                                         "ema_v": ema_v, "ema_signed": ema_signed})

                        # --- CSV row (write smoothed values) ---
                        log.writerow([f, tid, cname, x1, y1, x2, y2, conf_i,
                                      cx, cy, X_m, Y_m, ema_v, ema_signed])

                        # Trails
                        if tid not in trails:
                            trails[tid] = deque(maxlen=TRAIL_LEN)
                        trails[tid].append((cx, cy))
                        last_seen[tid] = 0
                        seen_now.add(tid)

                    # Age and prune old trails
                    for tid in list(last_seen.keys()):
                        if tid not in seen_now:
                            last_seen[tid] += 1
                            if last_seen[tid] > TRAIL_AGE_DROP:
                                last_seen.pop(tid, None)
                                trails.pop(tid, None)

                    # Draw boxes/labels once
                    labels = []
                    for i, box in enumerate(tracked.xyxy):
                        tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                        spd = kin_state.get(tid, {}).get("ema_v", 0.0)
                        vsg = kin_state.get(tid, {}).get("ema_signed", 0.0)
                        # show signed velocity; add ↑/↓ arrows for fun
                        arrow = "↑" if vsg > 0 else ("↓" if vsg < 0 else "·")
                        labels.append(f"disc #{tid} {spd:.1f} m/s  {arrow}{abs(vsg):.1f}")

                    frame = box_annotator.annotate(scene=frame, detections=tracked)
                    frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)

                # draw trails (after boxes so trails sit underneath labels)
                draw_trails(frame, trails)

            else:
                # naive tracker path (assign class by index; for precise mapping use IoU if needed)
                boxes_int = [list(map(int, b)) for b in xyxy.tolist()] if len(xyxy) else []
                tracks = tracker.update(boxes_int) if boxes_int else []
                current_ids = set()
                for j, (tid, box) in enumerate(tracks):
                    x1, y1, x2, y2 = map(int, box)
                    name = cls[j] if j < len(cls) else "disc"
                    conf_val = float(conf[j]) if j < len(conf) else 0.0

                    # Center, meters, speed
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    X_m, Y_m = img_to_meters(cx, cy)

                    now = f
                    prev = kin_state.get(tid)
                    if prev is None:
                        ema_v = 0.0
                        ema_signed = 0.0
                        kin_state[tid] = {"last_xy_m": (X_m, Y_m), "last_frame": now,
                                          "ema_v": ema_v, "ema_signed": ema_signed}
                    else:
                        px, py = prev["last_xy_m"]
                        dt = max(1, now - prev["last_frame"]) / FPS
                        
                        dx = X_m - px
                        dy = Y_m - py
                        step = float(np.hypot(dx, dy))

                        max_step = MAX_SPEED_MS * dt
                        if step > max_step and step > 0:
                            scale = max_step / step
                            dx *= scale
                            dy *= scale
                            X_m = px + dx
                            Y_m = py + dy
                            step = max_step

                        raw_speed = step / dt
                        signed_v  = float((dx * U_HAT[0] + dy * U_HAT[1]) / dt)

                        ema_v      = ALPHA_V * raw_speed + (1.0 - ALPHA_V) * prev["ema_v"]
                        ema_signed = ALPHA_V * signed_v  + (1.0 - ALPHA_V) * prev["ema_signed"]

                        prev.update({"last_xy_m": (X_m, Y_m), "last_frame": now,
                                     "ema_v": ema_v, "ema_signed": ema_signed})


                    # CSV
                    log.writerow([f, int(tid), name, x1, y1, x2, y2, conf_val,
                                  cx, cy, X_m, Y_m, ema_v, ema_signed])

                    # Draw
                    arrow = "↑" if ema_signed > 0 else ("↓" if ema_signed < 0 else "·")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"disc #{int(tid)} {ema_v:.1f} m/s {arrow}{abs(ema_signed):.1f}",
                                (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Trails
                    if tid not in trails:
                        trails[tid] = deque(maxlen=TRAIL_LEN)
                    trails[tid].append((cx, cy))
                    last_seen[tid] = 0
                    current_ids.add(tid)

                # age & prune
                for tid in list(last_seen.keys()):
                    if tid not in current_ids:
                        last_seen[tid] += 1
                        if last_seen[tid] > TRAIL_AGE_DROP:
                            last_seen.pop(tid, None)
                            trails.pop(tid, None)

                draw_trails(frame, trails)

            # HUD / debug
            cv2.putText(frame, f"f={f} det={len(xyxy)}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, f"f={f} det={len(xyxy)}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, f"kept={len(xyxy)}", (12, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frame, f"kept={len(xyxy)}", (12, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

            writer.write(frame)
            f += 1

    cap.release()
    writer.release()
    print(f"✅ Saved: {VIDEO_OUT} and {CSV_OUT}")
    if total > 0:
        print(f"Frames with detections (post-filter): {with_det}/{total} = {with_det/total:.1%}")

if __name__ == "__main__":
    main()

