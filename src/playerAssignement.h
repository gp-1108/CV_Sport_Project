/**
* @file playerAssignement.h
* @author Enrico D'Alberton ID number: 2093708
* @date ---
* @version 1.0
*/

#ifndef PLAYERASSIGNEMENT_H
#define PLAYERASSIGNEMENT_H

#include <opencv2/opencv.hpp>

/**
 * @brief this function generates the .txt files containing the coordinates of the players in the image and the .png file containing the colore mask of the players assigned to their team
 * @param originalImage: reference of the Mat object containing the original colored image (BGR format)
 * @param playerMask: reference of the Mat object containing the mask of the players (black pixels are background, non-black pixels are players) 
*/
void playerAssignement(cv::Mat& originalImage, cv::Mat& playerMask);

#endif