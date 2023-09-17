"""
  Author: Pietro Girotto
  ID: 2088245

  This file is meant to convert the dataset LV-MHP-v1 from the original of binary mask to the format of YOLO.
  The original dataset is composed of images and annotations. The annotations are binary masks of the objects
  in the image. The goal is to convert the binary masks to the format of YOLO, which is <class> <x1> <y1> <x2> <y2>
  where the coordinates are relative to the image size.

  The script is easily customizable to use different approaches to merge the contours of the binary mask.
  Have a look at the contours_utils.py file for available functions. It is as simple as changing the function
  name in the cu.<function_name> call.
  Not the cu.bin_mask_to_polys function, which is used to convert the binary mask to a list of polygons and is
  necessary for all the other functions.

  The flag DRAW can be used to visualize the result of the merging process, without saving the new dataset.

  For usage instructions, run the script with the -h flag.
"""

DRAW = False

import os
import numpy as np
from PIL import Image
import sys
from alive_progress import alive_bar
import cv2
import contours_utils as cu
import argparse

if __name__ == "__main__":
  # Parse the arguments
  parser = argparse.ArgumentParser(description="Convert the LV-MHP-v1 dataset from binary mask to YOLO format.")
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

  # Check if the dataset path exists
  if not os.path.exists(dataset_path):
    print("Dataset path does not exist!")
    sys.exit(1)

  if not is_draw and not (os.path.exists(new_dataset_path) 
          and os.path.exists(os.path.join(new_dataset_path, "images"))
          and os.path.exists(os.path.join(new_dataset_path, "labels"))):
    os.mkdir(new_dataset_path)
    os.mkdir(os.path.join(new_dataset_path, "images"))
    os.mkdir(os.path.join(new_dataset_path, "labels"))

  # Opening the file with the annotations
  with open(os.path.join(dataset_path, "images_annotations.txt"), "r") as f:
    lines = f.readlines()
    with alive_bar(len(lines)) as bar:
      for line in lines:
        line = line.strip().split(" ")
        image_path = os.path.join(dataset_path, "images", line[0])

        # Copy the image
        image = Image.open(image_path)

        cv2_image = cv2.imread(image_path)

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

        image_masks = line[1:]
        first_line = True
        if not is_draw:
          f = open(os.path.join(new_dataset_path, "labels", line[0].replace(".jpg", ".txt")), "w")
        for i, mask in enumerate(image_masks):
          polys = cu.bin_mask_to_polys(os.path.join(dataset_path, "annotations", mask))

          ################ SELECT A FUNCTION FROM HERE ####################
          # final_vertices = cu.angle_bins_approach(polys, bin_num=50)    #
          # final_vertices = cu.greatest_poly(polys)                      #
          # final_vertices = cu.all_vertices(polys)                       #
          #################################################################
          final_vertices = cu.all_vertices(polys)

          if is_draw:
            # Draw the contours
            cv2.polylines(cv2_image, np.array([final_vertices]), True, colors[i % len(colors)], 2)
            continue

          # Write the label in <class> <x1> <y1> <x2> <y2> format
          if not first_line:
            f.write("\n")
          first_line = False
          text = "0 "
          for coord in final_vertices:
            coord[0] = coord[0] / image.size[0]
            coord[1] = coord[1] / image.size[1]
            coord[0] = round(coord[0], 4)
            coord[1] = round(coord[1], 4)
            text += str(coord[0]) + " " + str(coord[1]) + " "
          text = text[:-1]
          f.write(text)
        if is_draw:
          cv2.imshow("image", cv2_image)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
        else:
          cv2.imwrite(os.path.join(new_dataset_path, "images", line[0]), cv2_image)
          f.close()
        bar()

  if is_draw:
    # If we are only drawing the result, we can exit here
    sys.exit(0)

  train_list = os.path.join(dataset_path, "train_list.txt")
  test_list = os.path.join(dataset_path, "test_list.txt")

  # Open the train and test list
  train_list = open(train_list, "r").readlines()
  test_list = open(test_list, "r").readlines()

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
  
  # Remove the old dataset structure
  os.rmdir(os.path.join(dataset_path, "images"))
  os.rmdir(os.path.join(dataset_path, "annotations"))

  # Adding the dataset yaml file
  with open(os.path.join(new_dataset_path, "dataset.yaml"), "w") as f:
    f.write("train: <define_path>/train\n")
    f.write("val: <define_path>/test\n")
    f.write("nc: 1\n")
    f.write("names: ['person']\n")