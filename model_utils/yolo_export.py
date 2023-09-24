"""
  Author: Pietro Girotto
  ID: 2088245

  This script is used to export the trained model to ONNX format.
  Simply pass the path to the model.pt file as argument.
"""
from ultralytics import YOLO
import sys

model = YOLO("yolov8m-seg.pt")
model = YOLO(sys.argv[1])  # model.pt path

model.export(format="onnx",
             dynamic=False,
             opset=11,
             imgsz=640,
             half=False,
             simplify=False)