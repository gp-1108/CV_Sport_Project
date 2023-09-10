import sys
import os

import cv2

train_list = open(sys.argv[1], "r").read().splitlines()
test_list = open(sys.argv[2], "r").read().splitlines()

dataset_path = sys.argv[3]

train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

if not os.path.exists(train_path):
  os.mkdir(train_path)
  os.mkdir(os.path.join(train_path, "images"))
  os.mkdir(os.path.join(train_path, "labels"))
  os.mkdir(test_path)
  os.mkdir(os.path.join(test_path, "images"))
  os.mkdir(os.path.join(test_path, "labels"))

for image_name in train_list:
  image_path = os.path.join(dataset_path, "images", image_name)
  label_path = os.path.join(dataset_path, "labels", image_name.replace(".jpg", ".txt"))

  # Move image and label in the train folder
  os.rename(image_path, os.path.join(train_path, "images", image_name))
  os.rename(label_path, os.path.join(train_path, "labels", image_name.replace(".jpg", ".txt")))

for image_name in test_list:
  image_path = os.path.join(dataset_path, "images", image_name)
  label_path = os.path.join(dataset_path, "labels", image_name.replace(".jpg", ".txt"))

  # Move image and label in the test folder
  os.rename(image_path, os.path.join(test_path, "images", image_name))
  os.rename(label_path, os.path.join(test_path, "labels", image_name.replace(".jpg", ".txt")))