import cv2
import os

video_path = "/Users/jameselsner/Desktop/Projects/ddc-vision/Videos/GOTW.mp4"
output_folder = "frames_output"
frame_rate = 1  # Extract 1 frame per second

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * frame_rate)

count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1
    count += 1

cap.release()
print(f"✅ Saved {saved} frames to '{output_folder}'")
