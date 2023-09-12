/**
* @file postProcessing.h
* @author Enrico D'Alberton ID number: 2093708
* @date ---
* @version 1.0
*/

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <opencv2/opencv.hpp>

/**
 * @brief this function processes the output obtained from the neural network to improve the player segmentation
 * @param originalImage: reference of the Mat object containing the original colored image (BGR format)
 * @param playerMask: reference of the Mat object containing the mask of the players (black pixels are background, non-black pixels are players) 
 * @return Mat object which represents the processed mask of the players
*/
void postProcessing(cv::Mat& originalImage, cv::Mat& playerMask);

#endif