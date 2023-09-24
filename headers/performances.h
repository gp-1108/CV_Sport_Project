/**
* @file performances.h
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
 * @brief function that computes the metrics required for the players localization (mAP, mean Average Precision) for the predicted bounding boxes
 * and for the images segmentation (mIoU, mean Intersection over Union). It also writes a file which will contain all the metrics computed for each image and the overall results
 * @param folderPredictionPath: path to the folder that contains the files in which are stored all the predictions obtained during the execution of the program
 * @param folderGroundTruthPath: path to the folder that contains the files in which are stored all the ground truth files
*/
void computeMetrics(std::string folderPredictionPath, std::string folderGroundTruthPath);

/**
 * @brief function that retreives the predicted bounding boxes saved in a certain files and returns them, paired with the corresponding label and confidence value, in a vector
 * @param filePath: path of the file in which the information of the bounding boxes is stored
 * @param inverted: flag that tells if the predicted bounding boxes will be considered with the inverted label or not
 * @return vector of tuples (rectangle, label, confidence) containing the information of each predicted bounding box
*/
std::vector<std::tuple<cv::Rect, int, double>> getBoundingBoxesFromPredictionsFile(std::string filePath, bool inverted);

/**
 * @brief function that retreives the ground truth bounding boxes saved in a certain files and returns them, paired with the corresponding label, in a vector
 * @param filePath: path of the file in which the information of the bounding boxes is stored
 * @return vector of tuples (rectangle, label) containing the information of each bounding box
*/
std::vector<std::tuple<cv::Rect, int>> getBoundingBoxesFromGroundTruthFile(std::string filePath);

/**
 * @brief function that computes the IoU (Intersection over Union) for two bounding boxes
 * @param prediction: predicted bounding box
 * @param groundTruth: true bounding box
 * @return double value corresponding to the IoU of the two bounding boxes
*/
double intersectionOverUnionBoundingBox(const cv::Rect prediction, const cv::Rect groundTruth);

/**
 * @brief function that computes the IoU (Intersection over Union) for segmentation masks of a certain class
 * @param prediction: predicted mask
 * @param groundTruth: true mask
 * @param label: value corresponding to the class label
 * @return double value corresponding to the IoU of the segmentation masks of a certain class
*/
double intersectionOverUnionSegmentation(const cv::Mat prediction, const cv::Mat groundTruth, int label);

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
double mAPComputation(std::vector<std::tuple<cv::Rect, int, double>> predictions, std::vector<std::tuple<cv::Rect, int>> truth);

#endif