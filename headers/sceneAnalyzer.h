/**
* @file sceneAnalyzer.h
* @author Enrico D'Alberton ID number: 2093708
*/

#ifndef SCENEANALYZER_H
#define SCENEANALYZER_H

#include "yolo.h"
#include <opencv2/opencv.hpp>

/**
 * @brief this function removes the confidence value from the bounding boxes file
 * @param output_folder_path: string containing the path of the folder where the bb.txt file is stored
 * @param file_name: string containing the name of the file analyzed
*/
void txtCleanUp(const std::string& output_folder_path, const std::string& file_name);

/**
 * @brief this function runs the entire algorithm on the image passed as parameter
 * @param yolo: reference of the Yolov8Seg object
 * @param output_folder_path: string containing the path of the folder where the output files will be saved
 * @param file_name: string containing the name of the file to be analyzed
*/
void sceneAnalyzer(Yolov8Seg& yolo, const std::string& output_folder_path, const std::string& file_name);

#endif