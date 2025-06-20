from ultralytics import YOLO

# Select model
model = YOLO("yolo11n.pt")

# Select image size (640 recommended)
imgsz = 640

model.export(format="ncnn", imgsz=imgsz)