import cv2, os

video_path = "/Users/jameselsner/Desktop/Escape/Games/13Sep2025/LaterGames.mp4"
output_folder = "frames_output"
seconds_between = 2.0

os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video.")

next_capture_time = 0.0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # current position in milliseconds
    t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if t_sec >= next_capture_time:
        filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1
        next_capture_time += seconds_between

cap.release()
print(f"✅ Saved {saved} frames to '{output_folder}'")
