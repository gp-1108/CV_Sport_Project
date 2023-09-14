/**
* @file fieldSegmentation.h
* @author Enrico D'Alberton ID number: 2093708
* @date ---
* @version 1.0
*/

#include "fieldSegmentation.h"
#include "yolo.h"
#include "postProcessingYolo.h"
#include "playerAssignement.h"
#include <opencv2/opencv.hpp>

void sceneAnalyzer(Yolov8Seg& yolo, const std::string& output_folder_path, const std::string& file_name) {
  // Reading the image
  cv::Mat original_image = cv::imread(file_name);

  // Field segmentation
  cv::Mat field_mask;
  field_mask = fieldDetectionAndSegmentation(original_image);
  // Turn field_mask from BGR to binary 
  //cv::cvtColor(field_mask, field_mask, cv::COLOR_BGR2GRAY);
  //cv::threshold(field_mask, field_mask, 0, 3, cv::THRESH_BINARY); //TODO controllare

  // Running the segmentation
  cv::Mat processed_mask;
  yolo.runSegmentation(original_image, processed_mask);
  processed_mask = postProcessingYolo(processed_mask); //TODO modifica direttamente la reference
  assignToTeams(output_folder_path, file_name, original_image, processed_mask, field_mask);
}