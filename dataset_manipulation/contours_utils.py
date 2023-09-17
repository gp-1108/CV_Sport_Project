"""
  Author: Pietro Girotto
  ID: 2088245

  This file is a collection of functions used to manipulate the contours of the binary masks.
"""

import math
import cv2
import numpy as np

def bin_mask_to_polys(mask_path):
  """
    This function is used to convert a binary mask to a list of polygons.
    The binary mask is read from the path specified in the argument.
    The function uses the OpenCV library to find the contours of the binary mask, 
    a threshold is applied to the binary mask to obtain a binary image.

    Arguments:
    ----------

    mask_path: str -> path to the binary mask

    Returns:
    --------

    all_polys: [[[x1, y1], [x2, y2], ...], ...] list -> list of polygons, each polygon is a list of vertices
  """

  cv2_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
  _, binary_image = cv2.threshold(cv2_mask, 1, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Computing a list of all vertices
  all_polys = []
  for contour in contours:
    poly = []
    for point in contour:
      point = point[0]
      poly.append([point[0], point[1]])
    all_polys.append(poly)

  return all_polys
    
def angle_bins_approach(polys, bin_num=50):
  """
    This function is used to merge vertices from different unconnected components of the binary mask
    into a single polygon. The approach used is to first compute the centroid of each unconnected component,
    the compute the centroid of all centroids (weighted by the number of vertices of each component).
    Then the vertices are divided into bins based on the angle from the centroid of all centroids to the vertex.
    The vertex with the maximum distance from the centroid of all centroids is kept for each bin.
    Fine tuning of the bin number is needed to obtain the best results, for the lv_mhp dataset
    50 bins seems to work well enough.

    Arguments:
    ----------
    
    polys: [[[x1, y1], [x2, y2], ...], ...] list -> list of polygons, each polygon is a list of vertices
    bin_num: int -> number of bins to divide the vertices

    Returns:
    --------

    result: [[x1, y1], [x2, y2], ...] list -> list of vertices of the merged polygon
  """

  # Computing all vertices and all centroids 
  # for each disconnected component
  all_vertices = []
  centroids = []
  for poly in polys:
    centroid = [0, 0]
    for point in poly:
      centroid[0] += point[0]
      centroid[1] += point[1]
      all_vertices.append(point)
    centroid[0] /= len(poly)
    centroid[1] /= len(poly)
    centroids.append((centroid, len(poly)))

  # Computing the final centroid
  final_centroid = [0, 0]
  all_length = 0
  for pair in centroids:
    centroid = pair[0]
    final_centroid[0] += centroid[0] * pair[1]
    final_centroid[1] += centroid[1] * pair[1]
    all_length += pair[1]
  final_centroid[0] /= all_length
  final_centroid[1] /= all_length

  # Calculate angles from centroid to vertices and store them along with the vertices.
  angles_with_vertices = []
  for vertex in all_vertices:
    angle = math.atan2(vertex[1] - centroid[1], vertex[0] - centroid[0])
    if angle < 0:
      angle += 2 * math.pi  # Convert angles to [0, 2*pi] range.
    angles_with_vertices.append((angle, vertex))

  # Sort vertices based on angles.
  angles_with_vertices.sort(key=lambda x: x[0])

  # Initialize bins.
  bins = [[] for _ in range(bin_num)]

  # Populate bins with vertices.
  for angle, vertex in angles_with_vertices:
    bin_index = int(angle / (2 * math.pi / bin_num))
    bins[bin_index].append(vertex)

  # Initialize the result list to store merged vertices.
  result = []

  # Process each bin and keep the furthest vertex.
  for bin_vertices in bins:
      if bin_vertices:
        furthest_vertex = max(bin_vertices, key=lambda vertex: math.dist(centroid, vertex))
        result.append(furthest_vertex)

  # Sort the result in clockwise order.
  result.sort(key=lambda vertex: math.atan2(vertex[1] - centroid[1], vertex[0] - centroid[0]))

  # Change to standard list format.
  result = [[vertex[0], vertex[1]] for vertex in result]

  return result

def greatest_poly(polys):
  """
    This function returns the polygon with the greatest area from a list of polygons specified
    by their vertices.

    Arguments:
    ----------

    polys: [[[x1, y1], [x2, y2], ...], ...] list -> list of polygons, each polygon is a list of vertices

    Returns:
    --------

    max_poly: [[x1, y1], [x2, y2], ...] list -> list of vertices of the polygon with the greatest area
  """
  max_area = 0
  max_poly = None
  for poly in polys:
    area = cv2.contourArea(np.array([poly]))
    if area > max_area:
      max_area = area
      max_poly = poly
  
  return max_poly

def all_vertices(polys):
  """
    This function returns all the vertices from a list of polygons specified
    by their vertices.

    Arguments:
    ----------

    polys: [[[x1, y1], [x2, y2], ...], ...] list -> list of polygons, each polygon is a list of vertices

    Returns:
    --------

    all_vertices: [[x1, y1], [x2, y2], ...] list -> list of all vertices
  """
  all_vertices = []
  for poly in polys:
    for vertex in poly:
      all_vertices.append(vertex)
  
  return all_vertices