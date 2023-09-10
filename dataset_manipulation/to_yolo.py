"""
  This file is meant to convert the dataset LV-MHP-v1 from the original of binary mask to the format of YOLO.
"""

import os
import numpy as np
from PIL import Image
import sys
from alive_progress import alive_bar
import cv2
from rdp import rdp
import random
import math

def find_closest(poly1, poly2):
  min_dist = 10000000
  poly1_point = None
  poly2_point = None
  for point1 in poly1:
    for point2 in poly2:
      dist = np.linalg.norm(np.array(point1) - np.array(point2))
      if dist < min_dist:
        min_dist = dist
        poly1_point = point1
        poly2_point = point2
  return poly1_point, poly2_point, min_dist

def calculate_centroid(polygon):
  # Calculate the centroid of a polygon represented as a list of points.
  x_sum = sum(point[0] for point in polygon)
  y_sum = sum(point[1] for point in polygon)
  centroid_x = x_sum / len(polygon)
  centroid_y = y_sum / len(polygon)
  return (centroid_x, centroid_y)

def pairwise_distance(polygon1, polygon2):
  # Calculate the Euclidean distance between centroids of two polygons.
  centroid1 = calculate_centroid(polygon1)
  centroid2 = calculate_centroid(polygon2)
  return math.dist(centroid1, centroid2)

def sort_polygons_by_proximity(polygons):
  sorted_polygons = [polygons[0]]  # Start with the first polygon.
  remaining_polygons = polygons[1:]

  while remaining_polygons:
    closest_polygon = min(
      remaining_polygons,
      key=lambda polygon: pairwise_distance(sorted_polygons[-1], polygon)
    )
    sorted_polygons.append(closest_polygon)
    remaining_polygons.remove(closest_polygon)

  return sorted_polygons


if __name__ == "__main__":
  # Define the paths
  dataset_path = sys.argv[1]
  new_dataset_path = sys.argv[2]

  # Check if the dataset path exists
  if not os.path.exists(dataset_path):
    print("Dataset path does not exist!")
    sys.exit(1)

  if not (os.path.exists(new_dataset_path) 
          and os.path.exists(os.path.join(new_dataset_path, "images"))
          and os.path.exists(os.path.join(new_dataset_path, "labels"))):
    os.mkdir(new_dataset_path)
    os.mkdir(os.path.join(new_dataset_path, "images"))
    os.mkdir(os.path.join(new_dataset_path, "labels"))

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
        with open(os.path.join(new_dataset_path, "labels", line[0].split(".")[0] + ".txt"), "w") as f:
          first_line = True
          for i, mask in enumerate(image_masks):
            mask_path = os.path.join(dataset_path, "annotations", mask)
            mask = Image.open(mask_path)
            mask = np.array(mask).astype(np.uint8)
            coords = np.where(mask > 0)

            cv2_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_image = cv2.threshold(cv2_mask, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_contours = []
            for contour in contours:
              total_contours += contour.tolist()
            contours = total_contours
            # cv2.polylines(cv2_image, np.array([contours], np.int32), True, colors[i], 2)
            # cv2.imshow("image", cv2_image)
            # cv2.waitKey(0)

            # Write the label in <class> <x1> <y1> <x2> <y2> format
            if not first_line:
              f.write("\n")
            first_line = False
            text = "0 "
            for coord in total_contours:
              coord = coord[0]
              coord[0] = coord[0] / image.size[0]
              coord[1] = coord[1] / image.size[1]
              coord[0] = round(coord[0], 4)
              coord[1] = round(coord[1], 4)
              text += str(coord[0]) + " " + str(coord[1]) + " "
            text = text[:-1]
            f.write(text)
            continue

            for i in range(len(coords[0])):
              x_rel = coords[1][i] / image.size[0]
              y_rel = coords[0][i] / image.size[1]
              # Round to 3 decimals
              x_rel = round(x_rel, 3)
              y_rel = round(y_rel, 3)
              formatted_coords.append([x_rel, y_rel])
            
            # Finding the convex hull
            # hull = ConvexHull(formatted_coords)
            # sorted_coords = [formatted_coords[i] for i in hull.vertices]
            # formatted_coords = sorted_coords

            # Dropping randomly 30% of the points
            formatted_coords = random.sample(formatted_coords, 1000)
            print(len(formatted_coords))
            formatted_coords = rdp(formatted_coords, epsilon=0.1)
            print(len(formatted_coords))

            
            draw_coords = []
            for coord in formatted_coords:
              draw_coords.append([coord[0] * image.size[0], coord[1] * image.size[1]])
            # Draw the polygon
            cv2.polylines(cv2_image, np.array([draw_coords], dtype=np.int32), True, (0, 0, 128), 2)
            cv2.imshow("image", cv2_image)
            cv2.waitKey(0)

          cv2.imwrite(os.path.join(new_dataset_path, "images", line[0]), cv2_image)
          f.close()
        bar()