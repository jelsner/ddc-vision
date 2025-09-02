# ddc_predict_discs_with_tracking.py
import os, cv2, csv, base64, requests, numpy as np
from collections import deque

# ====== CONFIG ======
API_KEY   = "wNNtaVXlX79MsMdYXtEc"
MODEL_ID  = "ddc-discs-5vdna/7"    # RF-DETR v7
VIDEO_IN  = "/Users/jameselsner/Desktop/Projects/ddc-vision/Videos/Jim_Vergil.mov"
VIDEO_OUT = "predicted_tracked3.mp4"
CSV_OUT   = "tracks.csv"

CONFIDENCE   = 0.21
FRAME_STRIDE = 1
RESIZE_WIDTH = 1536
TOP_K        = 2

# Trail drawing
TRAIL_LEN       = 32   # number of recent points to keep per track
TRAIL_MIN_STEP  = 2    # draw every Nth segment to avoid overdraw
TRAIL_THICK_MAX = 5    # starting line thickness
TRAIL_THICK_MIN = 1    # ending line thickness
TRAIL_COLOR     = (0, 255, 255)  # BGR (yellow); box is green, trail is yellow
TRAIL_AGE_DROP  = 90   # remove trail if track hasn't been seen for N frames
# ====================

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
        self.next_id = 1; self.tracks = {}; self.iou_thresh = iou_thresh; self.max_age = max_age
    @staticmethod
    def iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        aa=(a[2]-a[0])*(a[3]-a[1]); bb=(b[2]-b[0])*(b[3]-b[1])
        return inter / (aa+bb-inter+1e-6)
    def update(self, boxes):
        for tid in list(self.tracks):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age: del self.tracks[tid]
        assigned={}
        for b in boxes:
            best_tid,best_iou=None,0.0
            for tid,info in self.tracks.items():
                i=self.iou(b,info["bbox"])
                if i>best_iou: best_tid,best_iou=tid,i
            if best_iou>=self.iou_thresh:
                self.tracks[best_tid]={"bbox":b,"age":0}; assigned[tuple(b)]=best_tid
            else:
                tid=self.next_id; self.next_id+=1
                self.tracks[tid]={"bbox":b,"age":0}; assigned[tuple(b)]=tid
        return [(assigned[tuple(b)], b) for b in boxes]

_last_scale = 1.0
def rf_infer(frame_bgr):
    """Roboflow Hosted Inference → JSON."""
    global _last_scale
    if RESIZE_WIDTH is not None:
        h,w = frame_bgr.shape[:2]
        _last_scale = RESIZE_WIDTH/float(w)
        frame_bgr = cv2.resize(frame_bgr, (RESIZE_WIDTH, int(h*_last_scale)))
    else:
        _last_scale = 1.0
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok: return {"predictions":[]}
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    r = requests.post(
        f"https://detect.roboflow.com/{MODEL_ID}",
        params={"api_key": API_KEY, "confidence": CONFIDENCE},
        data=img_b64,
        headers={"Content-Type":"application/x-www-form-urlencoded"}
    )
    return r.json() if r.ok else {"predictions":[]}

def to_xyxy_conf_cls(preds):
    xyxy, conf, cls = [], [], []
    for d in preds:
        x,y,w,h = d["x"], d["y"], d["width"], d["height"]
        xyxy.append([x-w/2, y-h/2, x+w/2, y+h/2])
        conf.append(d.get("confidence",0.0))
        cls.append((d.get("class") or d.get("label") or "disc").lower())
    if not xyxy:
        return np.empty((0,4),np.float32), np.empty((0,),np.float32), []
    return np.array(xyxy,np.float32), np.array(conf,np.float32), cls

def main():
    assert os.path.exists(VIDEO_IN), f"Video not found: {VIDEO_IN}"
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened(): raise RuntimeError("OpenCV failed to open the video.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W,H))

    # Tracker
    if USE_BYTETRACK and HAVE_SUPERVISION:
        tracker = BYTETracker(BYTETrackerArgs(
            track_thresh=BT_TRACK_THRESH, match_thresh=BT_MATCH_THRESH,
            track_buffer=BT_TRACK_BUFFER, frame_rate=BT_FRAME_RATE))
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
        log.writerow(["frame","track_id","class","x1","y1","x2","y2","conf"])

        f = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if f % FRAME_STRIDE != 0:
                writer.write(frame); f += 1; continue

            # ==== INFERENCE ====
            data = rf_infer(frame)
            preds = data.get("predictions", [])
            xyxy, conf, cls = to_xyxy_conf_cls(preds)
            if _last_scale != 1.0 and len(xyxy): 
                xyxy /= _last_scale

            # ==== FILTERING (apply BEFORE counting and TOP_K) ====
            def min_conf_for_class(cl, base=CONFIDENCE):
                name = (cl or "disc").lower()
                if "yellow" in name: return max(0.18, base - 0.02)
                if "red"    in name: return max(0.20, base)
                return base

            filtered_boxes, filtered_conf, filtered_cls = [], [], []
            for b, c, cl in zip(xyxy, conf, cls):
                x1,y1,x2,y2 = map(int, b)
                w, h = x2 - x1, y2 - y1
                ar = max(w, h) / float(min(w, h) + 1e-6)
                frac = (w * h) / float(W * H)

                # Shape gating
                if ar > 2.0: 
                    continue

                # Confidence gating (per class)
                if c < min_conf_for_class(cl):
                    continue

                filtered_boxes.append(b); filtered_conf.append(c); filtered_cls.append(cl)

            xyxy = np.array(filtered_boxes, np.float32) if filtered_boxes else np.empty((0,4), np.float32)
            conf = np.array(filtered_conf,  np.float32) if filtered_conf else np.empty((0,), np.float32)
            cls  = filtered_cls

            # Keep at most TOP_K after filtering
            if len(conf) > TOP_K:
                idx = np.argsort(-conf)[:TOP_K]
                xyxy, conf = xyxy[idx], conf[idx]
                cls = [cls[i] for i in idx]

            # Count AFTER filtering so stats match what we draw
            total += 1
            if len(xyxy) > 0: 
                with_det += 1

            # ==== TRACKING ====
            if use_bt:
                det = Detections(xyxy=xyxy, confidence=conf) if len(xyxy) else Detections.empty()
                tracked = tracker.update_with_detections(det)
                labels = []
                seen_now = set()
                if len(tracked) > 0:
                    # Attach original class by IoU for CSV only
                    classes_for_tracked = []
                    for tbox in tracked.xyxy:
                        best_iou, best_c = 0.0, "disc"
                        for i,b in enumerate(xyxy):
                            x1=max(int(tbox[0]),int(b[0])); y1=max(int(tbox[1]),int(b[1]))
                            x2=min(int(tbox[2]),int(b[2])); y2=min(int(tbox[3]),int(b[3]))
                            inter=max(0,x2-x1)*max(0,y2-y1)
                            aa=(int(tbox[2]-tbox[0]))*(int(tbox[3]-tbox[1]))
                            bb=(int(b[2]-b[0]))*(int(b[3]-b[1]))
                            iou = inter / (aa+bb-inter+1e-6)
                            if iou>best_iou:
                                best_iou, best_c = iou, (cls[i] if i < len(cls) else "disc")
                        classes_for_tracked.append(best_c)

                    for i, box in enumerate(tracked.xyxy):
                        tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                        c   = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                        x1,y1,x2,y2 = map(int, box.tolist())
                        labels.append(f"disc #{tid} {c:.2f}")
                        log.writerow([f, tid, classes_for_tracked[i], x1,y1,x2,y2, c])

                        # ----- update trail -----
                        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                        if tid not in trails: trails[tid] = deque(maxlen=TRAIL_LEN)
                        trails[tid].append((cx, cy))
                        last_seen[tid] = 0
                        seen_now.add(tid)

                    # age and prune old trails
                    for tid in list(last_seen.keys()):
                        if tid not in seen_now:
                            last_seen[tid] += 1
                            if last_seen[tid] > TRAIL_AGE_DROP:
                                last_seen.pop(tid, None)
                                trails.pop(tid, None)

                    # draw boxes/labels
                    frame = box_annotator.annotate(scene=frame, detections=tracked)
                    frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)

                # draw trails (after boxes so trails sit underneath labels)
                draw_trails(frame, trails)

            else:
                # naive tracker path (also gets trails)
                boxes_int = [list(map(int, b)) for b in xyxy.tolist()] if len(xyxy) else []
                tracks = tracker.update(boxes_int) if boxes_int else []
                current_ids = set()
                for tid, box in tracks:
                    x1,y1,x2,y2 = map(int, box)
                    log.writerow([f, int(tid), "disc", x1,y1,x2,y2, 0.0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame, f"disc #{int(tid)}", (x1, max(0, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    # trail update
                    cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                    if tid not in trails: trails[tid] = deque(maxlen=TRAIL_LEN)
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
                # draw trails
                draw_trails(frame, trails)

            # HUD / debug
            cv2.putText(frame, f"f={f} det={len(xyxy)}", (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
            cv2.putText(frame, f"f={f} det={len(xyxy)}", (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
            cv2.putText(frame, f"kept={len(xyxy)}", (12,54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, f"kept={len(xyxy)}", (12,54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1)

            writer.write(frame)
            f += 1

    cap.release(); writer.release()
    print(f"✅ Saved: {VIDEO_OUT} and {CSV_OUT}")
    if total>0:
        print(f"Frames with detections (post-filter): {with_det}/{total} = {with_det/total:.1%}")

def draw_trails(frame, trails):
    """Draw fading polylines for each track's recent centers."""
    for tid, pts in trails.items():
        if len(pts) < 2: 
            continue
        # Fade thickness from head (latest) to tail (oldest)
        for i in range(len(pts)-1, 0, -TRAIL_MIN_STEP):
            p1 = pts[i]
            p2 = pts[i-1]
            # thickness decreases towards the tail
            t = int(np.interp(i, [1, len(pts)-1],
                              [TRAIL_THICK_MIN, TRAIL_THICK_MAX]))
            cv2.line(frame, p1, p2, TRAIL_COLOR, thickness=t)
        # draw a small head dot
        cv2.circle(frame, pts[-1], 3, (0, 255, 0), -1)

if __name__ == "__main__":
    main()
