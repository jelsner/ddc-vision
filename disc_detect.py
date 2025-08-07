import cv2
import numpy as np
from PIL import Image

# === Load image ===
image_path = "/Users/jameselsner/Desktop/Projects/ddc-vision/Pictures/Disc_Red.png"
pil_img = Image.open(image_path).convert("RGB")
img = np.array(pil_img, dtype=np.uint8)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# === Convert to HSV ===
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# === Red, Orange, Yellow color masks ===
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
lower_orange = np.array([11, 100, 100])
upper_orange = np.array([25, 255, 255])
lower_yellow = np.array([26, 100, 100])
upper_yellow = np.array([35, 255, 255])

mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

combined_mask = cv2.bitwise_or(mask_red1, mask_red2)
combined_mask = cv2.bitwise_or(combined_mask, mask_orange)
combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)

# === Morphological smoothing ===
kernel = np.ones((5, 5), np.uint8)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

# === Apply mask ===
masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=combined_mask)

# === Grayscale & blur ===
gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.medianBlur(gray, 5)

# === Crop tightly around disc ===
y_offset, x_offset = 300, 200
crop_h, crop_w = 300, 300
cropped = gray_blurred[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]

# === Hough Circle Transform ===
circles = cv2.HoughCircles(
    cropped,
    method=cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=100,
    param2=25,         # More conservative
    minRadius=30,
    maxRadius=150
)

# === Draw only the largest circle ===
if circles is not None:
    circles = np.uint16(np.around(circles))

    # Find circle closest to center of crop
    crop_center = (crop_w // 2, crop_h // 2)
    def dist_to_center(circle):
        return (circle[0] - crop_center[0])**2 + (circle[1] - crop_center[1])**2

    best_circle = min(circles[0], key=dist_to_center)

    center = (best_circle[0] + x_offset, best_circle[1] + y_offset)
    radius = best_circle[2]

    cv2.circle(img_bgr, center, radius, (0, 255, 0), 2)
    cv2.circle(img_bgr, center, 2, (0, 0, 255), 3)

    print(f"Detected 1 circle at {center} with radius {radius}")
else:
    print("No circles detected.")
    print("All detected circles (x, y, r):")
    print(circles[0])


# === Show final result ===
cv2.imshow("Refined Disc Detection", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
