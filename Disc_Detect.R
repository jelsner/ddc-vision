library(reticulate)

use_python("/Users/jameselsner/miniforge3/envs/ddc-ai/bin/python", required = TRUE)

cv2 <- import("cv2")
np <- import("numpy")
PIL <- import("PIL.Image")

# Load image and convert to RGB
pil_img <- PIL$open("Pictures/DDC_Discs.jpg")$convert("RGB")

# Get size tuple and convert to R vector
size <- r_to_py(pil_img$size)
width <- size[1]
height <- size[2]

# Convert image to byte buffer
img_bytes <- pil_img$tobytes()

# Convert to NumPy array and reshape
img_np <- np$frombuffer(img_bytes, dtype = "uint8")
expected_size <- height * width * 3
actual_size <- py_to_r(img_np$size)

if (actual_size != expected_size) {
  stop(paste("Size mismatch:", actual_size, "vs expected", expected_size))
}

img_np <- r_to_py(img_np)$reshape(c(height, width, 3L))

# Convert RGB to BGR for OpenCV
img_bgr <- cv2$cvtColor(img_np, cv2$COLOR_RGB2BGR)


# Convert to grayscale
gray <- cv2$cvtColor(img, cv2$COLOR_BGR2GRAY)

# Apply blur
gray_blurred <- cv2$medianBlur(gray, 5L)

# Detect circles
circles <- cv2$HoughCircles(
  image = gray_blurred,
  method = cv2$HOUGH_GRADIENT,
  dp = 1L,
  minDist = 50L,
  param1 = 100L,
  param2 = 30L,
  minRadius = 10L,
  maxRadius = 100L
)

# Draw the circles
if (!is.null(circles)) {
  circles <- np$uint16(np$around(circles))
  for (i in 0:(dim(circles[[1]])[0] - 1)) {
    x <- circles[[1]][[i]][[0]]
    y <- circles[[1]][[i]][[1]]
    r <- circles[[1]][[i]][[2]]
    cv2$circle(img, center = tuple(x, y), radius = r, color = tuple(0L, 255L, 0L), thickness = 2L)
    cv2$circle(img, center = tuple(x, y), radius = 2L, color = tuple(0L, 0L, 255L), thickness = 3L)
  }
} else {
  cat("No circles detected.\n")
}

# Show the image
cv2$imshow("Detected Discs", img)
cv2$waitKey(0L)
cv2$destroyAllWindows()
