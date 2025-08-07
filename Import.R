library(reticulate)

# Set the environment explicitly
use_python("/Users/jameselsner/miniforge3/envs/ddc-ai/bin/python", required = TRUE)

# Try importing OpenCV
cv2 <- import("cv2")
cv2[["__version__"]]
