import os
import csv
import cv2
import re

# === CONFIG ===
VIDEO_IN  = "/Users/jameselsner/Desktop/Escape/Rallies/ForPrediction/Rally1.mov"
CSV_IN    = "Videos/Annotated/track_corrected_rally1.csv"    # from rf_video_export_flex.py
VIDEO_OUT = "Videos/Annotated/annotated_rally1c.mp4"

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.8     # tuned for 4K; try 1.2–2.2 as needed
THICK      = 4
BOX_THICK  = 3
PAD_Y      = 10      # vertical padding for label
# =============

# BGR colors
RED     = (0,   0, 255)
YELLOW  = (0, 255, 255)
BLUE    = (255, 0,   0)
WHITE   = (255, 255, 255)
GREEN   = (0, 255, 0)
BLACK   = (0, 0, 0)

def to_xyxy(x, y, w, h):
    x1 = int(x - w/2.0); y1 = int(y - h/2.0)
    x2 = int(x + w/2.0); y2 = int(y + h/2.0)
    return x1, y1, x2, y2

def class_to_color(name: str):
    """Return (box_color, text_color) based on class name."""
    n = (name or "").lower()

    # discs
    if "disc_red" in n:
        return RED, RED
    if "disc_yellow" in n:
        return YELLOW, YELLOW
    if n == "disc":
        return GREEN, GREEN  # generic disc, fallback

    # players (team A/B detection by substring)
    # accepts forms like "playerA_1", "lead_playerA_2", etc.
    if re.search(r"playera", n):
        return BLUE, BLUE
    if re.search(r"playerb", n):
        return WHITE, WHITE

    # default
    return GREEN, GREEN

def load_preds(csv_path):
    """
    Returns: dict[int_frame] -> list of dicts with keys:
    class, conf, x, y, w, h
    """
    by_frame = {}
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        # Expected headers: frame,time_s,class,class_id,x,y,width,height,confidence,x1,y1,x2,y2
        for row in r:
            try:
                fidx = int(float(row["frame"]))
            except:
                continue
            try:
                det = {
                    "cls": row.get("class", "obj"),
                    "conf": float(row.get("confidence") or 0.0),
                    "x": float(row["x"]), "y": float(row["y"]),
                    "w": float(row["width"]), "h": float(row["height"])
                }
            except Exception:
                # skip malformed rows
                continue
            by_frame.setdefault(fidx, []).append(det)
    return by_frame

def draw_label(img, text, org, color):
    """
    Draw readable label: dark outline + colored text.
    org is baseline-left (like putText).
    """
    x, y = org
    # shadow/outline
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, BLACK, THICK+2, lineType=cv2.LINE_AA)
    # main
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, color, THICK, lineType=cv2.LINE_AA)

def main():
    assert os.path.exists(VIDEO_IN), f"Video not found: {VIDEO_IN}"
    assert os.path.exists(CSV_IN),   f"CSV not found: {CSV_IN}"

    preds = load_preds(CSV_IN)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open the input video.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    f = 0
    total_boxes = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = preds.get(f, [])

        for d in dets:
            x1, y1, x2, y2 = to_xyxy(d["x"], d["y"], d["w"], d["h"])
            box_color, text_color = class_to_color(d["cls"])
            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICK)

            # label
            label = f'{d["cls"]} {d["conf"]:.2f}'
            # baseline under/above box with padding
            label_x = x1
            label_y = max(0, y1 - PAD_Y)
            draw_label(frame, label, (label_x, label_y), text_color)

            total_boxes += 1

        # small HUD
        hud = f"frame={f} det={len(dets)}"
        draw_label(frame, hud, (12, 50), WHITE)

        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    print(f"✅ Saved: {VIDEO_OUT} (drew {total_boxes} boxes over {f} frames)")

if __name__ == "__main__":
    main()
