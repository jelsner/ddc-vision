import os
import cv2
import csv
import json
import base64
import requests
import numpy as np

# ========= CONFIG: set these =========
API_KEY   = "wNNtaVXlX79MsMdYXtEc"             # Roboflow > Settings > API Keys
MODEL_ID  = "ddc-discs-5vdna/2"             # From Deploy tab, e.g. "<project_slug>/<version>"
VIDEO_IN  = "/Users/jameselsner/Desktop/Projects/ddc-vision/Videos/Augie_Highlight_Volley.mp4"
VIDEO_OUT = "predicted_tracked.mp4"
CSV_OUT   = "tracks.csv"

CONFIDENCE   = 0.16      # 0.15–0.30; lower = more detections (helps tracking)
FRAME_STRIDE = 1        # process every Nth frame (2 = ~15 fps to API)
RESIZE_WIDTH = 1280      # downscale width before sending to API; None = original (4K is heavy)

# ByteTrack tuning (sports-friendly). If FRAME_STRIDE=2, use frame_rate=15.
BT_FRAME_RATE  = 30 if FRAME_STRIDE == 1 else 15
BT_TRACK_THRESH = max(CONFIDENCE, 0.18)    # accept lower conf to keep tracks alive
BT_MATCH_THRESH = 0.88    # stricter association to reduce ID swaps
BT_TRACK_BUFFER = 60     # ~1.5s at 30fps (or ~3s at 15fps)

# ====================================

# Try ByteTrack via supervision; fallback to a tiny naive IoU tracker
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

class NaiveTracker:
    """Minimal IoU-based tracker (fallback if supervision is unavailable)."""
    def __init__(self, iou_thresh=0.3, max_age=15):
        self.next_id = 1
        self.tracks = {}  # tid -> {bbox, age}
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    @staticmethod
    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, boxes):
        assigned = {}
        # age & drop stale
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]
        # greedy assign by IoU
        for b in boxes:
            best_tid, best_iou = None, 0.0
            for tid, info in self.tracks.items():
                i = self.iou(b, info["bbox"])
                if i > best_iou:
                    best_tid, best_iou = tid, i
            if best_iou >= self.iou_thresh:
                self.tracks[best_tid] = {"bbox": b, "age": 0}
                assigned[tuple(b)] = best_tid
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"bbox": b, "age": 0}
                assigned[tuple(b)] = tid
        return [(assigned[tuple(b)], b) for b in boxes]

_last_scale = 1.0  # set per-frame when resizing for inference

def roboflow_predict(frame_bgr):
    """Send frame (optionally downscaled) to Roboflow Hosted Inference, return JSON."""
    global _last_scale
    if RESIZE_WIDTH is not None:
        h, w = frame_bgr.shape[:2]
        _last_scale = RESIZE_WIDTH / float(w)
        frame_bgr = cv2.resize(frame_bgr, (RESIZE_WIDTH, int(h*_last_scale)), interpolation=cv2.INTER_LINEAR)
    else:
        _last_scale = 1.0

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return {"predictions": [], "image": {"width": frame_bgr.shape[1], "height": frame_bgr.shape[0]}}

    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    r = requests.post(
        f"https://detect.roboflow.com/{MODEL_ID}",
        params={"api_key": API_KEY, "confidence": CONFIDENCE},
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        return r.json()
    except Exception:
        return {"predictions": []}

def preds_to_xyxy(preds):
    """Convert Roboflow center-format boxes to [x1,y1,x2,y2] and confidences."""
    xyxy, conf = [], []
    for d in preds:
        x, y, w, h = d["x"], d["y"], d["width"], d["height"]
        xyxy.append([x - w/2, y - h/2, x + w/2, y + h/2])
        conf.append(d.get("confidence", 0.0))
    if not xyxy:
        return np.empty((0,4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.array(xyxy, dtype=np.float32), np.array(conf, dtype=np.float32)

def disc_color_name(frame_bgr, box_xyxy):
    """Classify disc color (red / yellow) by HSV pixel count within the box."""
    x1, y1, x2, y2 = [max(0, int(v)) for v in box_xyxy]
    x2, y2 = min(x2, frame_bgr.shape[1]-1), min(y2, frame_bgr.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return "disc"
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "disc"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Red (two hue ranges) and Yellow
    red1 = cv2.inRange(hsv, (0, 120, 80),  (10, 255, 255))
    red2 = cv2.inRange(hsv, (170,120, 80), (180,255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    yellow_mask = cv2.inRange(hsv, (18, 120, 80), (35, 255, 255))

    red_score = int(np.sum(red_mask) // 255)
    yellow_score = int(np.sum(yellow_mask) // 255)
    if max(red_score, yellow_score) < 50:  # not enough colored pixels
        return "disc"
    return "red" if red_score >= yellow_score else "yellow"

def main():
    assert os.path.exists(VIDEO_IN), f"Video not found: {VIDEO_IN}"
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open the video.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Opened video: {W}x{H} @ {FPS:.2f}fps")

    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    if USE_BYTETRACK and HAVE_SUPERVISION:
        args = BYTETrackerArgs(
            track_thresh=BT_TRACK_THRESH,
            match_thresh=BT_MATCH_THRESH,
            track_buffer=BT_TRACK_BUFFER,
            frame_rate=BT_FRAME_RATE
        )
        tracker = BYTETracker(args)
        palette = ColorPalette.default()
        box_annotator = BoxAnnotator(color=palette, thickness=2)
        label_annotator = LabelAnnotator(color=palette)
        print("Tracker: ByteTrack (supervision)")
    else:
        tracker = NaiveTracker(iou_thresh=0.35, max_age=45)
        print("Tracker: Naive IoU (fallback)")

    with open(CSV_OUT, "w", newline="") as fcsv:
        writer_csv = csv.writer(fcsv)
        writer_csv.writerow(["frame", "track_id", "color", "x1", "y1", "x2", "y2", "conf"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STRIDE != 0:
                # keep timing alignment; write unmodified frame
                cv2.putText(frame, f"f={frame_idx} det=skip", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
                cv2.putText(frame, f"f={frame_idx} det=skip", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
                writer.write(frame)
                frame_idx += 1
                continue

            rf_out = roboflow_predict(frame)
            preds = rf_out.get("predictions", [])
            xyxy, conf = preds_to_xyxy(preds)

            # rescale boxes back to original 4K coords if we downscaled for inference
            if _last_scale != 1.0 and len(xyxy):
                xyxy = xyxy / _last_scale

            # keep at most the top-2 boxes (you expect at most 2 discs)
            if len(conf) > 2:
                top_idx = np.argsort(-conf)[:2]
                xyxy = xyxy[top_idx]
                conf = conf[top_idx]
                
                # --- after you have: xyxy, conf = preds_to_xyxy(preds); and the optional rescale ---

# 1) Keep only plausible red/yellow discs (color + size/shape gates)
def is_red_or_yellow(frame_bgr, box_xyxy, min_pixels=120):
    x1,y1,x2,y2 = [max(0, int(v)) for v in box_xyxy]
    x2, y2 = min(x2, frame_bgr.shape[1]-1), min(y2, frame_bgr.shape[0]-1)
    if x2 <= x1 or y2 <= y1: return False
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0: return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0,120,80),  (10,255,255))
    red2 = cv2.inRange(hsv, (170,120,80),(180,255,255))
    yellow = cv2.inRange(hsv, (18,120,80),(35,255,255))
    red = cv2.bitwise_or(red1, red2)
    score = int(np.sum(red)//255) + int(np.sum(yellow)//255)
    return score >= min_pixels

def plausible_disc(box, img_w, img_h, min_frac=0.0035, max_frac=0.06, max_ar=2.0):
    x1,y1,x2,y2 = box
    w = max(1, x2-x1); h = max(1, y2-y1)
    ar = max(w,h) / float(min(w,h))
    area = w*h; frac = area / float(img_w*img_h + 1e-6)
    return (ar <= max_ar) and (min_frac <= frac <= max_frac)

# Apply the gates
W, H = frame.shape[1], frame.shape[0]
keep_boxes, keep_conf = [], []
for b, c in zip(xyxy, conf):
    if is_red_or_yellow(frame, b) and plausible_disc(b, W, H):
        keep_boxes.append(b); keep_conf.append(c)

if keep_boxes:
    xyxy = np.array(keep_boxes, dtype=np.float32)
    conf = np.array(keep_conf, dtype=np.float32)
else:
    xyxy = np.empty((0,4), dtype=np.float32)
    conf = np.empty((0,), dtype=np.float32)

# 2) Keep at most the top-2 detections (there are ≤2 discs)
if len(conf) > 2:
    top_idx = np.argsort(-conf)[:2]
    xyxy = xyxy[top_idx]
    conf = conf[top_idx]

            if USE_BYTETRACK and HAVE_SUPERVISION:
                det = Detections(xyxy=xyxy, confidence=conf) if len(xyxy) else Detections.empty()
                tracked = tracker.update_with_detections(det)   # Detections with tracker_id
                labels = []
                if len(tracked) > 0:
                    for i, box in enumerate(tracked.xyxy):
                        tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                        c   = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                        color = disc_color_name(frame, box)
                        labels.append(f"{color} #{tid} {c:.2f}")
                        x1, y1, x2, y2 = map(int, box.tolist())
                        writer_csv.writerow([frame_idx, tid, color, x1, y1, x2, y2, c])

                    frame = box_annotator.annotate(scene=frame, detections=tracked)
                    frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)
            else:
                # naive tracker path
                boxes_int = [list(map(int, b)) for b in xyxy.tolist()] if len(xyxy) else []
                tracks = tracker.update(boxes_int) if boxes_int else []
                for tid, box in tracks:
                    x1, y1, x2, y2 = map(int, box)
                    color = disc_color_name(frame, box)
                    # draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{color} #{tid}", (x1, max(0, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    # best-effort conf (match to nearest det by IoU)
                    best_c = 0.0
                    for j, bb in enumerate(xyxy):
                        xx1, yy1, xx2, yy2 = map(int, bb)
                        ix1, iy1 = max(x1, xx1), max(y1, yy1)
                        ix2, iy2 = min(x2, xx2), min(y2, yy2)
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        area_a = (x2-x1)*(y2-y1)
                        area_b = (xx2-xx1)*(yy2-yy1)
                        iou = inter / (area_a + area_b - inter + 1e-6)
                        if iou > 0.5:
                            best_c = float(conf[j])
                            break
                    writer_csv.writerow([frame_idx, int(tid), color, x1, y1, x2, y2, best_c])

            # HUD
            cv2.putText(frame, f"f={frame_idx} det={len(xyxy)}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
            cv2.putText(frame, f"f={frame_idx} det={len(xyxy)}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"✅ Saved annotated video: {VIDEO_OUT}")
    print(f"✅ Saved tracks CSV: {CSV_OUT}")

if __name__ == "__main__":
    main()
