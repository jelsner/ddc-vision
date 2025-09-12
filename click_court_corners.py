import cv2
import os

# === CONFIG ===
VIDEO_IN = "/Users/jameselsner/Desktop/Projects/ddc-vision/Videos/Annotated/annotated_rally2_tracked.mp4"
FRAME_IDX = 100  # which frame to show
OUT_TXT   = "court_points.txt"
DISPLAY_WIDTH = 1280   # 960 max width for display window (scaled down)
# ==============

# Load video and grab one frame
cap = cv2.VideoCapture(VIDEO_IN)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Could not read frame from video.")

# Compute scale for display
scale = DISPLAY_WIDTH / frame.shape[1]
disp = cv2.resize(frame, (DISPLAY_WIDTH, int(frame.shape[0] * scale)))

points = []

def click_event(event, x, y, flags, param):
    global points, disp
    if event == cv2.EVENT_LBUTTONDOWN:
        # rescale click to original resolution
        orig_x, orig_y = int(x / scale), int(y / scale)
        points.append((orig_x, orig_y))
        # mark on the display image
        cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(disp, f"{len(points)}", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Select Corners", disp)

cv2.imshow("Select Corners", disp)
cv2.setMouseCallback("Select Corners", click_event)

print("Click the four court corners in order: top-left, top-right, bottom-right, bottom-left.")
print("Press 'q' when done.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or len(points) >= 4:
        break

cv2.destroyAllWindows()

if len(points) == 4:
    with open(OUT_TXT, "w") as f:
        for (x, y) in points:
            f.write(f"{x},{y}\n")
    print(f"✅ Saved 4 points to {OUT_TXT}: {points}")
else:
    print("⚠️ Did not capture 4 points.")
