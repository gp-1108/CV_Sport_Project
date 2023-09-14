/**
* @file postProcessing.cpp
* @author Enrico D'Alberton ID number: 2093708
* @date ---
* @version 1.0
*/
#include "postProcessingYolo.h"
#include <opencv2/opencv.hpp>

cv::Mat postProcessingYolo(cv::Mat& playerMask) {

  cv::Mat editedPlayerMask = cv::Mat::zeros(playerMask.size(), CV_8UC1);

  // Find bounding boxes of players
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(playerMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Find the bounding boxes of the players
  std::vector<cv::Rect> boundingBoxes;
  for (int i = 0; i < contours.size(); i++) {
    cv::Rect boundingBox = cv::boundingRect(contours[i]);
    boundingBoxes.push_back(boundingBox);
  }

  // Find the average area of the bounding boxes
  double averageArea = 0;
  for (int i = 0; i < boundingBoxes.size(); i++) {
    averageArea += boundingBoxes[i].area();
  }
  averageArea /= boundingBoxes.size();

  // If the area of each bounding box is less than 1/5 of the average area, remove it
  for (int i = 0; i < boundingBoxes.size(); i++) {
    if (boundingBoxes[i].area() < averageArea / 5) {
      boundingBoxes.erase(boundingBoxes.begin() + i);
      i--;
    }
  }

  for(int i = 1; i < 256; i++) {
    cv::Mat copyPlayerMask = playerMask.clone();
    bool found = false;
    for(int j = 0; j < playerMask.rows; j++) {
      for(int k = 0; k < playerMask.cols; k++) {
        if(playerMask.at<uchar>(j, k) == i) {
          found = true;
        }
        if(copyPlayerMask.at<uchar>(j, k) != i) {
          copyPlayerMask.at<uchar>(j, k) = 0;
        }
      }
    }
    if(found) {
      // Apply morphological operations to the mask to remove noise
      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
      cv::erode(copyPlayerMask, copyPlayerMask, kernel);
      cv::erode(copyPlayerMask, copyPlayerMask, kernel);
      cv::erode(copyPlayerMask, copyPlayerMask, kernel);
      cv::erode(copyPlayerMask, copyPlayerMask, kernel);
      cv::dilate(copyPlayerMask, copyPlayerMask, kernel);
      cv::dilate(copyPlayerMask, copyPlayerMask, kernel);
      cv::dilate(copyPlayerMask, copyPlayerMask, kernel);
      cv::dilate(copyPlayerMask, copyPlayerMask, kernel);
      editedPlayerMask = editedPlayerMask + copyPlayerMask;
    }
  }
  return editedPlayerMask;

}