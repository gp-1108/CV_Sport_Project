/**
* @file playerAssignement.h
* @author Enrico D'Alberton ID number: 2093708
* @date ---
* @version 1.0
*/

#ifndef PLAYERASSIGNEMENT_H
#define PLAYERASSIGNEMENT_H

#include <opencv2/opencv.hpp>

float silhouette(std::vector<std::tuple<int, int, int>> match, std::vector<std::vector<std::tuple<int, int, int>>> clusters, int k);

float silhouette(std::vector<std::tuple<int, int>> match, std::vector<std::vector<std::tuple<int, int>>> clusters, int k);

std::vector<std::vector<std::tuple<int, int, int>>> k_means(std::vector<std::tuple<int, int, int>> match, int k);

std::vector<std::vector<std::tuple<int,int>>> k_means(std::vector<std::tuple<int,int>> match, int k);

float findMax(float a, float b, float c, float d, float e, float f);

void localizePlayers(const cv::Mat& original_image, const cv::Mat& mask, std::vector<std::tuple<cv::Rect, cv::Mat, int>>& players);

void parseClusters(std::vector<std::vector<std::tuple<int, int, int>>> clusters, std::vector<std::tuple<int, int, int>> match, std::vector<int>& team_membership);

void parseClusters(std::vector<std::vector<std::tuple<int, int>>> clusters, std::vector<std::tuple<int, int>> match, std::vector<int>& team_membership);

void saveOutput(const std::string& output_folder_path, const std::string& file_name, const cv::Mat& RGB_mask, const cv::Mat& BN_mask, const std::vector<std::tuple<cv::Rect, cv::Mat, int>>& players, const std::vector<int>& team_membership);

void assignToTeams(const std::string& output_folder_path, std::string file_name, cv::Mat& original_image, cv::Mat& mask);

/**
 * @brief this function generates the .txt files containing the coordinates of the players in the image and the .png file containing the colore mask of the players assigned to their team
 * @param originalImage: reference of the Mat object containing the original colored image (BGR format)
 * @param playerMask: reference of the Mat object containing the mask of the players (black pixels are background, non-black pixels are players) 
*/
void playerAssignement(const std::string& model_path, const std::string& folder_path, const std::string& output_folder_path);

#endif