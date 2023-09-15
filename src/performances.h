/**
* @file performance.h
* @author Federico Gelain ID number: 2076737
* @date ---
* @version 1.0
*/

#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <vector>
#include <opencv2/highgui.hpp>
#include <string>

/**
 * @brief function that computes the metric for the players localization (mAP, mean Average Precision) for the predicted bounding boxes
 * @param folderPredictionPath: path to the folder that contains the files in which are stored the bounding boxes of each player predicted for each image
 * @param folderGroundTruthPath: path to the folder that contains the files in which are stored the bounding boxes of each player predicted for each image
*/
void playerLocalizationMetrics(std::string folderPredictionPath, std::string folderGroundTruthPath);

/**
 * @brief function that retreives the bounding boxes saved in a certain files and returns them, paired with the corresponding label, in a vector
 * @param filePath: path of the file in which the information of the bounding boxes is stored
 * @param inverted: flag that tells if the predicted bounding boxes will be considered with the inverted label or not
 * @return vector of pairs (label, rectangle) containing the information of each bounding box
*/
std::vector<std::pair<int, cv::Rect>> getBoundingBoxesFromFile(std::string filePath, bool inverted);

/**
 * @brief function that computes the IoU (Intersection over Union) for two bounding boxes
 * @param prediction: predicted bounding box
 * @param groundTruth: true bounding box
 * @return double value corresponding to the IoU of the two bounding boxes
*/
double intersectionOverUnion(const cv::Rect prediction, const cv::Rect groundTruth);

/**
 * @brief function that computes the AP (Average Precision) for a certain class
 * @param precision: vector of recall values associated with a certain class
 * @param recall: vector of recall values associated with a certain class
 * @return double value corresponding to the AP for a certain class
*/
double computeAP(std::vector<double> precision, std::vector<double> recall);

/**
 * @brief function that computes the mAP (mean Average Precision) of all classes
 * @param predictions: vector of pairs (label, rectangle) of all the predicted bounding boxes
 * @param truth: vector of pairs (label, rectangle) of all the ground truth bounding boxes
 * @return double value corresponding to the mAP of all classes
*/
double mAPComputation(std::vector<std::pair<int, cv::Rect>> predictions, std::vector<std::pair<int, cv::Rect>> truth);

#endif