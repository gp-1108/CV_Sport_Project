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
 * @param playerMask: reference of the Mat object containing the mask of the players (black pixels are background, non-black pixels are players) 
*/
void postProcessingYolo(cv::Mat& playerMask);

#endif