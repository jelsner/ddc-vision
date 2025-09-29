# rf_video_export_flex.py
import os, csv, json, urllib.request
from roboflow import Roboflow

# ===== CONFIG =====
API_KEY   = "XXXX"   # Roboflow > Settings > API Keys
PROJECT   = "ddc_game1_objectdetection-iqa0h"
VERSION   = "2"
VIDEO_IN  = "/Users/jameselsner/Desktop/Escape/Games/13Sep2025/G2_V1.mov"
FPS       = 29.97
PRED_TYPE = "batch-video"

JSON_OUT  = "Videos/Annotated/results_G2_V1.json"
CSV_OUT   = "Videos/Annotated/predictions_G2_V1.csv"
ANNO_OUT  = "Videos/Annotated/annotated_G2_V1.mp4"  # if a visualization URL is present
# ==================

def to_xyxy(x, y, w, h):
    x1 = x - w/2.0
    y1 = y - h/2.0
    x2 = x + w/2.0
    y2 = y + h/2.0
    return x1, y1, x2, y2

def main():
    assert os.path.exists(VIDEO_IN), f"Video not found: {VIDEO_IN}"

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(PROJECT)
    model = project.version(str(VERSION)).model

    # Kick off batch prediction
    job_id, signed_url, expire_time = model.predict_video(
        VIDEO_IN, fps=FPS, prediction_type=PRED_TYPE
    )
    results = model.poll_until_video_results(job_id)

    # Always save raw JSON (handy for debugging / re-parsing)
    with open(JSON_OUT, "w") as jf:
        json.dump(results, jf)
    print(f"📝 Saved raw JSON → {JSON_OUT}")

    # Try to download annotated video if link is present
    video_url = results.get("video") or results.get("visualization") or signed_url
    if video_url:
        try:
            os.makedirs(os.path.dirname(ANNO_OUT), exist_ok=True)
            print(f"⬇️  Downloading annotated video → {ANNO_OUT}")
            urllib.request.urlretrieve(video_url, ANNO_OUT)
            print("✅ Annotated video saved.")
        except Exception as e:
            print(f"⚠️ Could not download annotated video: {e}")

    # --- Parse predictions robustly across variants ---
    rows = []

    # Variant A: { "frame_offset":[...], "time_offset":[...], "<project>":[ {image:{}, predictions:[...]}, ...] }
    if PROJECT in results and isinstance(results[PROJECT], list):
        frames = results[PROJECT]
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

    # Variant B: results["predictions"] = [ {"frame_id":..., "predictions":[...]} , ... ]
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
        print("⚠️ Could not find predictions in results; inspect results JSON.")

    # Write CSV
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow([
            "frame","time_s","class","class_id",
            "x","y","width","height","confidence",
            "x1","y1","x2","y2"
        ])
        for r in rows:
            wri.writerow(r)

    print(f"📄 Saved {len(rows)} detections → {CSV_OUT}")

if __name__ == "__main__":
    main()
