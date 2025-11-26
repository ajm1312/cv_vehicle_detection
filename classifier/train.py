from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data='data.yaml',   # This file tells YOLO where your folders are
    epochs=50,          # How many times to cycle through the data
    imgsz=640,          # Image resolution
    batch=16,           # How many images to process at once
    name='vehicle_custom_model' # Saves results to runs/detect/vehicle_custom_model
)