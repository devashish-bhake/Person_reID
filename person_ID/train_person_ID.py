from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='/home/devashish/Desktop/work/Person_ReID/crowdhuman/data.yaml', epochs=100, batch=16, imgsz=416, verbose=True, device = 0)