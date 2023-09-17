"""
  Author: Pietro Girotto
  ID: 2088245


  This file is meant to be used to manipulate the oc_huma dataset to the format of YOLO.
  You can find it here: https://github.com/liruilong940607/OCHumanApi
  
  For usage instructions, run the script with the -h flag.
"""

import sys
import os
import json
from PIL import Image
from alive_progress import alive_bar
import argparse
import cv2
import random
import contours_utils as cu
import numpy as np

if __name__ == "__main__":
  # Parse the arguments
  parser = argparse.ArgumentParser(description="Convert the OCHuman dataset from binary mask to YOLO format.")
  parser.add_argument("--data", type=str, help="Root folder of the LV-MHP-V1 dataset.")
  parser.add_argument("--out", type=str, help="Root folder, where the new dataset will be saved.")
  parser.add_argument("--draw", type=bool, default=False, help="Show the result of the poly masks without saving the new dataset.")

  args = parser.parse_args()

  dataset_path = args.data
  new_dataset_path = args.out
  is_draw = args.draw

  if dataset_path is None or new_dataset_path is None:
    parser.print_help()
    sys.exit(1)

  json_path = os.path.join(dataset_path, "ochuman.json")

  # Create the new dataset folder
  if not is_draw and not os.path.exists(new_dataset_path):
    os.mkdir(new_dataset_path)
    os.mkdir(os.path.join(new_dataset_path, "images"))
    os.mkdir(os.path.join(new_dataset_path, "labels"))

  colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
    (128, 128, 128),
    (128, 0, 0),
    (128, 128, 0),
    (0, 128, 0),
    (128, 0, 128)
  ]


  # Load the json file
  with open(json_path, "r") as f:
    json_data = json.load(f)
    with alive_bar(len(json_data["images"])) as bar:
        for single_entry in json_data["images"]:
          annotations = single_entry["annotations"]
          image = cv2.imread(os.path.join(dataset_path, "images", single_entry["file_name"]))
          width, height = image.shape[1], image.shape[0]

          counter = 1
          output_text = ""
          for annotation in annotations:
            bbox = annotation["bbox"]
            
            seg = annotation["segms"]
            if seg == None:
              continue

            polys = []
            for single_seg in seg["outer"]:
              poly = []
              for i in range(0, len(single_seg) - 1, 2):
                x = int(single_seg[i])
                y = int(single_seg[i+1])
                poly.append([x, y])
              polys.append(poly)
            

            final_vertices = cu.angle_bins_approach(polys, bin_num=50)

            if is_draw:
              cv2.polylines(image, np.array([final_vertices]), True, colors[counter % len(colors)], 2)
            # Saving the polygon in yolo format
            temp_str = "0 "
            for vertex in final_vertices:
              x_rel = round(vertex[0] / width, 4)
              y_rel = round(vertex[1] / height, 4)
              temp_str += str(x_rel) + " " + str(y_rel) + " "
            temp_str = temp_str[:-1]
            temp_str += "\n"

            output_text += temp_str
            counter += 1
          if counter == 1: # No annotations
            continue
          output_text = output_text[:-1] # Removing the last \n

          if is_draw:
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
          else:
            # Saving the image and the label
            with open(os.path.join(new_dataset_path, "labels", single_entry["file_name"][:-4] + ".txt"), "w") as f:
              f.write(output_text)
            cv2.imwrite(os.path.join(new_dataset_path, "images", single_entry["file_name"]), image)
          bar()

if is_draw:
  exit(0)

# Dividing the dataset in train and test
train_path = os.path.join(new_dataset_path, "train")
test_path = os.path.join(new_dataset_path, "test")

if not os.path.exists(train_path):
  os.mkdir(train_path)
  os.mkdir(os.path.join(train_path, "images"))
  os.mkdir(os.path.join(train_path, "labels"))

if not os.path.exists(test_path):
  os.mkdir(test_path)
  os.mkdir(os.path.join(test_path, "images"))
  os.mkdir(os.path.join(test_path, "labels"))

images = os.listdir(os.path.join(new_dataset_path, "images"))
random.shuffle(images)
train_images = images[:int(len(images) * 0.8)]
test_images = images[int(len(images) * 0.8):]

for image in train_images:
  os.rename(os.path.join(new_dataset_path, "images", image), os.path.join(train_path, "images", image))
  os.rename(os.path.join(new_dataset_path, "labels", image[:-4] + ".txt"), os.path.join(train_path, "labels", image[:-4] + ".txt"))

for image in test_images:
  os.rename(os.path.join(new_dataset_path, "images", image), os.path.join(test_path, "images", image))
  os.rename(os.path.join(new_dataset_path, "labels", image[:-4] + ".txt"), os.path.join(test_path, "labels", image[:-4] + ".txt"))

# Deleting old images and labels folders in the new dataset
import shutil
shutil.rmtree(os.path.join(new_dataset_path, "images"))
shutil.rmtree(os.path.join(new_dataset_path, "labels"))

# Adding the dataset yaml file
with open(os.path.join(new_dataset_path, "dataset.yaml"), "w") as f:
  f.write("train: <define_path>/train\n")
  f.write("val: <define_path>/test\n")
  f.write("nc: 1\n")
  f.write("names: ['person']\n")