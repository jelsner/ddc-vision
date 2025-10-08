# rf_detector.py (resilient)
import os, csv, json, tempfile, shutil
import cv2
from roboflow import Roboflow

# ===== CONFIG =====
API_KEY   = "xxxxx"
WORKSPACE = None
PROJECT   = "ddc_game1_objectdetection-iqa0h"
VERSION   = "4"

VIDEO_IN  = "/Users/jameselsner/Desktop/Escape/Games/13Sep2025/Game4.mov"
FPS_PRIMARY = 29
PRED_TYPE   = "batch-video"
CSV_OUT     = "Videos/Annotated/predictions_Game4.csv"

# NEW: inference knobs (percent, not 0–1)
CONF_PCT    = 5     # 5% ≈ 0.05 confidence threshold
OVERLAP_PCT = 30    # NMS IoU threshold (typical 20–50)

def to_xyxy(x, y, w, h):
    x1 = x - w/2.0; y1 = y - h/2.0
    x2 = x + w/2.0; y2 = y + h/2.0
    return x1, y1, x2, y2

def reencode_to_mp4(src_path):
    """Re-encode any input video to H.264 MP4 to improve upload compatibility."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open: {src_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tmp_dir = tempfile.mkdtemp(prefix="rf_vid_")
    out_path = os.path.join(tmp_dir, "reencoded.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    ok_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        wr.write(frame)
        ok_frames += 1
    wr.release(); cap.release()

    if ok_frames == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("No frames written during re-encode.")
    return out_path, tmp_dir  # caller should cleanup tmp_dir

def get_model():
    rf = Roboflow(api_key=API_KEY)
    ws = rf.workspace(WORKSPACE) if WORKSPACE else rf.workspace()
    project = ws.project(PROJECT)
    return project.version(str(VERSION)).model

def upload_and_poll(model, video_path, fps, pred_type):
    # ❌ do NOT pass confidence / overlap here
    job_id, signed_url, expire_time = model.predict_video(
        video_path,
        fps=fps,
        prediction_type=pred_type
    )
    results = model.poll_until_video_results(job_id)
    return results, signed_url

def parse_and_write_csv(results, csv_out, project_slug):
    rows = []

    # Variant A: { "frame_offset":[...], "time_offset":[...], "<project>":[{...}] }
    if project_slug in results and isinstance(results[project_slug], list):
        frames = results[project_slug]
        frame_offsets = results.get("frame_offset", [])
        time_offsets  = results.get("time_offset", [])
        for i, fobj in enumerate(frames):
            preds = fobj.get("predictions", []) or []
            frame_idx = frame_offsets[i] if i < len(frame_offsets) else i
            t_sec     = time_offsets[i]  if i < len(time_offsets)  else None
            for det in preds:
                cls_name = det.get("class") or det.get("label")
                cls_id   = det.get("class_id")
                conf     = det.get("confidence")
                x        = det.get("x"); y = det.get("y")
                w        = det.get("width"); h = det.get("height")
                x1,y1,x2,y2 = to_xyxy(float(x), float(y), float(w), float(h))
                rows.append([frame_idx, t_sec, cls_name, cls_id, x, y, w, h, conf, x1, y1, x2, y2])

    # Variant B: results["predictions"] = [...]
    elif isinstance(results.get("predictions"), list):
        for fobj in results["predictions"]:
            frame_idx = fobj.get("frame_id", fobj.get("frame"))
            t_sec     = fobj.get("time", None)
            preds     = fobj.get("predictions", []) or []
            for det in preds:
                cls_name = det.get("class") or det.get("label")
                cls_id   = det.get("class_id")
                conf     = det.get("confidence")
                x        = det.get("x"); y = det.get("y")
                w        = det.get("width"); h = det.get("height")
                x1,y1,x2,y2 = to_xyxy(float(x), float(y), float(w), float(h))
                rows.append([frame_idx, t_sec, cls_name, cls_id, x, y, w, h, conf, x1, y1, x2, y2])

    # Variant C: results["output"]["predictions"] ...
    elif isinstance(results.get("output", {}).get("predictions"), list):
        for fobj in results["output"]["predictions"]:
            frame_idx = fobj.get("frame_id", fobj.get("frame"))
            t_sec     = fobj.get("time", None)
            preds     = fobj.get("predictions", []) or []
            for det in preds:
                cls_name = det.get("class") or det.get("label")
                cls_id   = det.get("class_id")
                conf     = det.get("confidence")
                x        = det.get("x"); y = det.get("y")
                w        = det.get("width"); h = det.get("height")
                x1,y1,x2,y2 = to_xyxy(float(x), float(y), float(w), float(h))
                rows.append([frame_idx, t_sec, cls_name, cls_id, x, y, w, h, conf, x1, y1, x2, y2])

    else:
        print("⚠️ Could not find predictions in results; dump the dict to inspect.")

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow([
            "frame","time_s","class","class_id",
            "x","y","width","height","confidence",
            "x1","y1","x2","y2"
        ])
        for r in rows:
            wri.writerow(r)

    print(f"📄 Saved {len(rows)} detections → {csv_out}")

def main():
    assert os.path.exists(VIDEO_IN), f"Video not found: {VIDEO_IN}"
    print("VIDEO_IN:", os.path.abspath(VIDEO_IN), f"({os.path.getsize(VIDEO_IN)/1e6:.1f} MB)")

    # Probe video locally
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError("OpenCV can't open your input video.")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Probe: {W}x{H} @ {src_fps:.2f} fps, ~{frames} frames")

    model = get_model()

    # Attempt 1: original file, primary params
    try:
        print("➡️  Uploading original file...")
        results, signed_url = upload_and_poll(model, VIDEO_IN, FPS_PRIMARY, PRED_TYPE)
    except Exception as e:
        msg = str(e)
        print("First attempt failed:", msg)

        # Common “BAD REQUEST” recovery: re-encode to H.264 MP4 and retry safer params
        print("🔁 Re-encoding to H.264 MP4 and retrying with fps=29…")
        tmp_path, tmp_dir = reencode_to_mp4(VIDEO_IN)
        try:
            results, signed_url = upload_and_poll(model, tmp_path, 29.97, "batch-video")
        finally:
            # Clean temp files
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Optional: save raw JSON for inspection (disabled by default)
    # with open("results_debug.json", "w") as jf:
    #     json.dump(results, jf)
    # print("📝 Saved raw JSON → results_debug.json")

    parse_and_write_csv(results, CSV_OUT, PROJECT)

if __name__ == "__main__":
    main()
