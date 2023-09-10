from ultralytics import YOLO
import sys
import os
from alive_progress import alive_bar

model = YOLO("yolov8m-seg.pt")
model = YOLO(sys.argv[1])  # model.pt path

dataset_path = sys.argv[2]


images_paths = []
for file_name in os.listdir(dataset_path):
  if file_name.endswith(".jpg") or file_name.endswith(".png"):
    image_path = os.path.join(dataset_path, file_name)
    images_paths.append(image_path)

with alive_bar(len(images_paths)) as bar:
  for image_path in images_paths:
    model.predict(source=image_path,
                  show=False,
                  save=True,
                  hide_labels=False,
                  save_txt=False,
                  conf=0.3,
                  hide_conf=False,
                  save_crop=False,
                  line_thickness=2)
    bar()