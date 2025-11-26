from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='vehicle_custom_model'
)