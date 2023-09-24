/**
* @file fieldSegmentation.h
* @author Federico Gelain ID number: 2076737
*/

#ifndef FIELDSEGMENTATION_H
#define FIELDSEGMENTATION_H

#include <opencv2/highgui/highgui.hpp>

/**
 * @brief main function that, given a sport image, determines where the field is located and return a mask where field and background are distinguished
 * @param fieldImage: original fieldImage, in BGR format, from which the field has to be detected and segmented
 * @return Mat object which represents a mask where black is the background, while green is the field
*/
cv::Mat fieldDetectionAndSegmentation(const cv::Mat fieldImage);

/**
 * @brief this function returns all the candidates colors for the field in the sport image
 * @param fieldImage: original fieldImage, in BGR format, from which to extract the candidate colors
 * @param mask: mask that indicates the region of interest of the image to calculate the colors
 * @param l: number of peaks to consier for each channel in the choice of the colors
 * @return vector of pairs where for each color an integer value is associated (see sortCandidateColors function below for more details of what it'll be used for)
*/
std::vector<std::pair<cv::Vec3b, int>> computeCandidateColors(cv::Mat fieldImage, cv::Mat mask, int l);

/**
 * @brief this function sorts the candidate colors in decreasing order based on how many pixels in the image are at minimum distance between them (i.e. compute for each
 * pixel in the image, within the mask, the closest candidate color and keep track of how many pixels will be associated with each candidate, storing said value as the integer
 * in each pair)
 * @param fieldImage: original fieldImage, in BGR format, to determine which are the most present candidate colors in the image
 * @param mask: mask that indicates the region of interest of the image to calculate the frequencies
 * @param candidateColors: array of pairs (candidateColor, frequency of candidate Color in the image). The second value, initially set to 0, is computed in this function
*/
void sortCandidateColors(const cv::Mat fieldImage, cv::Mat mask, std::vector<std::pair<cv::Vec3b, int>>& candidateColors);

/**
 * @brief this function returns the image where each pixel classified as field is assigned the value 3, 0 otherwise
 * @param fieldImage: original fieldImage, in BGR format
 * @param candidateColors: array of pairs (candidateColor, frequency of candidate Color in the image)
 * @param distanceThreshold: value which determines if a certain pixels is considered part of the field or not (based on its intensity Euclidean distance from the field color)
 * @param areaThreshold: value to decide if a certain candidate can be considered the field color or not
 * @return Mat where each pixel classified as field is assigned the value 3, 0 otherwise
*/
cv::Mat computeFieldMask(const cv::Mat fieldImage, std::vector<std::pair<cv::Vec3b, int>> candidateColors, int distanceThreshold, double areaThreshold);

/**
 * @brief some post processing ideas to improve the result given by just comparing the color of pixels compared to the one of the field
 * @param fieldImage: BGR image which comes as the result of segmentation considering only the intensity distance between pixels and the field color
*/
void fieldPostProcessing(cv::Mat& fieldImage);


#endif
