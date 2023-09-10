from ultralytics import YOLO
import sys

model = YOLO("yolov8m-seg.pt")
model = YOLO(sys.argv[1])  # model.pt path

model.export(format="onnx",
             imgsz=640,
             half=False,
             simplify=False)