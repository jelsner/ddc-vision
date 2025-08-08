from ultralytics import YOLO

# Load base model (can use 'yolov8n.pt' for faster training)
model = YOLO('yolov8n.pt')

# Train with your dataset
model.train(
    data='/Users/jameselsner/Desktop/Projects/ddc-vision/ddc-discs/data.yaml',
    epochs=50,
    imgsz=640,
    batch=4
)


#50 epochs completed in 0.110 hours.
#Optimizer stripped from /Users/jameselsner/runs/detect/train2/weights/last.pt, 6.2MB
#Optimizer stripped from /Users/jameselsner/runs/detect/train2/weights/best.pt, 6.2MB

#run
model = YOLO('/Users/jameselsner/runs/detect/train2/weights/best.pt')
results = model('/Users/jameselsner/Desktop/Projects/ddc-vision/Pictures/DDC_Discs.jpg', save=True)
