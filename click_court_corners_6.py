# click_court_corners_6_from_video.py
import os
import cv2
import json
import numpy as np
from datetime import datetime

# === CONFIG ===
VIDEO_IN       = "/Users/jameselsner/Desktop/Escape/Rallies/ForPrediction/Rally2.mp4"
FRAME_IDX      = 604        # which frame to show
OUT_TXT        = "court_points.txt"
OUT_JSON       = "calibration_6pts.json"
OUT_H_TXT      = "H.txt"
DISPLAY_WIDTH  = 1280       # max display width (window is scaled; saved pts are full-res)
# ==============

# World geometry (meters)
COURT_SIZE   = 13.0
GAP_BETWEEN  = 17.0

# Far court (top-left origin)
FAR_TL = (0.0, 0.0)
FAR_TR = (COURT_SIZE, 0.0)
FAR_BR = (COURT_SIZE, COURT_SIZE)
FAR_BL = (0.0, COURT_SIZE)

# Near-court "front" edge (closest to camera) is at y = -(GAP + COURT_SIZE)
NEAR_FRONT_Y = -(GAP_BETWEEN + COURT_SIZE)
NEAR_FL = (0.0,       NEAR_FRONT_Y)   # near front-left
NEAR_FR = (COURT_SIZE, NEAR_FRONT_Y)   # near front-right

WORLD_PTS_6 = np.array(
    [FAR_TL, FAR_TR, FAR_BR, FAR_BL, NEAR_FL, NEAR_FR],
    dtype=np.float32
)

INSTRUCTIONS = [
    "Click 6 points in this order:",
    "1: FAR Top-Left",
    "2: FAR Top-Right",
    "3: FAR Bottom-Right",
    "4: FAR Bottom-Left",
    "5: NEAR Front-Left (closest to camera)",
    "6: NEAR Front-Right (closest to camera)",
    "",
    "Keys:  [r] reset   [ENTER] save   [q] quit without saving"
]

def grab_frame(video_path: str, frame_idx: int):
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open the video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read frame from video.")
    return frame

def fit_to_width(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img.copy(), 1.0
    scale = max_w / float(w)
    new_wh = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_wh, interpolation=cv2.INTER_AREA), scale

def draw_ui(canvas, clicks_scaled):
    # Title/instructions
    y = 24
    for line in INSTRUCTIONS:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
        y += 26

    # Draw clicked points + indices
    for i, (sx, sy) in enumerate(clicks_scaled):
        cv2.circle(canvas, (int(sx), int(sy)), 6, (0,255,255), -1, cv2.LINE_AA)
        tag = str(i+1)
        cv2.putText(canvas, tag, (int(sx)+8, int(sy)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, tag, (int(sx)+8, int(sy)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)

def save_outputs(img_pts_fullres, H):
    # Save TXT (human-friendly)
    with open(OUT_TXT, "w") as f:
        f.write("# Court calibration points (pixel -> meters)\n")
        f.write("# Order: FAR TL, FAR TR, FAR BR, FAR BL, NEAR FL, NEAR FR\n\n")

        f.write("PIXEL_POINTS (x,y) in image pixels:\n")
        for i, (x, y) in enumerate(img_pts_fullres.tolist(), start=1):
            f.write(f"{i}: {x:.3f}, {y:.3f}\n")
        f.write("\nWORLD_POINTS (X_m,Y_m) in meters:\n")
        for i, (X, Y) in enumerate(WORLD_PTS_6.tolist(), start=1):
            f.write(f"{i}: {X:.3f}, {Y:.3f}\n")

    # Save JSON (data record)
    payload = {
        "video_in": VIDEO_IN,
        "frame_idx": FRAME_IDX,
        "timestamp": datetime.now().isoformat(),
        "pixel_points": img_pts_fullres.tolist(),
        "world_points": WORLD_PTS_6.tolist(),
        "notes": "Order: FAR TL, FAR TR, FAR BR, FAR BL, NEAR FL, NEAR FR"
    }
    with open(OUT_JSON, "w") as jf:
        json.dump(payload, jf, indent=2)

    # Save H.txt
    np.savetxt(OUT_H_TXT, H)

    print(f"✅ Saved: {OUT_TXT}")
    print(f"✅ Saved: {OUT_JSON}")
    print(f"✅ Saved: {OUT_H_TXT} (3x3 homography px→meters)")

def main():
    frame_full = grab_frame(VIDEO_IN, FRAME_IDX)       # full-res frame
    disp, scale = fit_to_width(frame_full, DISPLAY_WIDTH)

    win = "Click 6 Court Points (from video)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    clicks_full = []   # full-res pixel coords
    clicks_scaled = [] # scaled for display only

    def on_mouse(event, x, y, flags, param):
        nonlocal clicks_full, clicks_scaled
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks_full) < 6:
            # map scaled → full
            X = x / max(scale, 1e-9)
            Y = y / max(scale, 1e-9)
            clicks_full.append((float(X), float(Y)))
            clicks_scaled.append((float(x), float(y)))

    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = disp.copy()
        draw_ui(canvas, clicks_scaled)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("Quit without saving.")
            break

        if key == ord('r'):
            clicks_full.clear()
            clicks_scaled.clear()

        if key in (13, 10):  # ENTER
            if len(clicks_full) != 6:
                print(f"Need 6 points, you have {len(clicks_full)}.")
                continue

            IMG_PTS_6 = np.array(clicks_full, dtype=np.float32)
            H, mask = cv2.findHomography(IMG_PTS_6, WORLD_PTS_6, method=cv2.RANSAC)
            if H is None:
                print("Homography failed. Try clicking more carefully.")
                continue

            save_outputs(IMG_PTS_6, H)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
