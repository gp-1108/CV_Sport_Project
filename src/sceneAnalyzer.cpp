/**
* @file sceneAnalyzer.cpp
* @author Enrico D'Alberton ID number: 2093708
*/

#include "../headers/fieldSegmentation.h"
#include "../headers/yolo.h"
#include "../headers/postProcessingYolo.h"
#include "../headers/playerAssignement.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>


void txtCleanUp(const std::string& output_folder_path, const std::string& file_name) {

  // Txt file path
  std::string img_name = file_name.substr(file_name.find_last_of("/") + 1);
  img_name.erase(img_name.length() - 4);

  std::string bb_txt_path = output_folder_path + "/Masks/" + img_name + "_bb.txt";

    // Open the file for reading
    std::ifstream inputFile(bb_txt_path);

    // Read the content line by line into a vector
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(inputFile, line)) {
      // Fine the last space
      size_t lastSpace = line.find_last_of(" ");
      // Substring from the beginning to the last space
      line = line.substr(0, lastSpace);
      lines.push_back(line);
    }

    // Close the input file
    inputFile.close();

    // Open the same file for writing (truncate it)
    std::ofstream outputFile(bb_txt_path);
    if (!outputFile.is_open()) {
      std::cerr << "Failed to open the output file." << std::endl;
      return;
    }

    // Write the modified content back to the file
    for (const std::string& modifiedLine : lines) {
      outputFile << modifiedLine << std::endl;
    }

    // Close the output file
    outputFile.close();

}

void sceneAnalyzer(Yolov8Seg& yolo, const std::string& output_folder_path, const std::string& file_name) {
  // Reading the image
  cv::Mat original_image = cv::imread(file_name);

  // Field segmentation
  cv::Mat field_mask;
  field_mask = fieldDetectionAndSegmentation(original_image);

  // Running the segmentation
  cv::Mat processed_mask;
  yolo.runSegmentation(original_image, processed_mask);
  postProcessingYolo(processed_mask);
  assignToTeams(output_folder_path, file_name, original_image, processed_mask, field_mask);

}