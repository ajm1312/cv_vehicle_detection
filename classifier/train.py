from ultralytics import YOLO

model = YOLO('yolo11s.pt')

results = model.train(
    data='config.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='model',
)