from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model.train(
    data='config.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='model',
    cls=4.0,
    dropout=0.1
)