/**
* @file playerAssignement.cpp
* @author Enrico D'Alberton ID number: 2093708
* @date ---
* @version 1.0
*/
#include "playerAssignement.h"
#include "postProcessingYolo.h"
#include "yolo.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <tuple>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

float silhouette(std::vector<std::tuple<int, int, int>> match, std::vector<std::vector<std::tuple<int, int, int>>> clusters, int k) {
    float silhouette = 0;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < clusters[i].size(); j++) {
            float a = 0, b = std::numeric_limits<float>::max(); // Initialize b with a large value

            for (int l = 0; l < clusters[i].size(); l++) {
                if (l != j) {
                    float dist = pow(std::get<0>(clusters[i][j]) - std::get<0>(clusters[i][l]), 2) +
                                 pow(std::get<1>(clusters[i][j]) - std::get<1>(clusters[i][l]), 2) +
                                 pow(std::get<2>(clusters[i][j]) - std::get<2>(clusters[i][l]), 2);
                    a += dist;
                }
            }
            a = a / (clusters[i].size() - 1); // Average distance within cluster i

            for (int l = 0; l < k; l++) {
                if (l == i) continue;
                for (int m = 0; m < clusters[l].size(); m++) {
                    float dist = pow(std::get<0>(clusters[i][j]) - std::get<0>(clusters[l][m]), 2) +
                                 pow(std::get<1>(clusters[i][j]) - std::get<1>(clusters[l][m]), 2) +
                                 pow(std::get<2>(clusters[i][j]) - std::get<2>(clusters[l][m]), 2);
                    b = std::min(b, dist); // Find the smallest distance to other clusters
                }
            }

            float s; // Silhouette value for the data point
            if(clusters[i].size() == 1) {
              s = 0;
            } else {
              s = (b - a) / std::max(a, b);
            }

            silhouette += s;
        }
    }

    silhouette /= match.size(); // Average silhouette value across all data points
    return silhouette;
}

float silhouette(std::vector<std::tuple<int, int>> match, std::vector<std::vector<std::tuple<int, int>>> clusters, int k) {
  float silhouette = 0;

  for(int i = 0; i < k; i++) {
    for(int j = 0; j < clusters[i].size(); j++) {
      float a = 0, b = std::numeric_limits<float>::max(); // Initialize b with a large value

      for(int l = 0; l < clusters[i].size(); l++) {
        if(l != j) {
          float dist = pow(std::get<0>(clusters[i][j]) - std::get<0>(clusters[i][l]), 2) + pow(std::get<1>(clusters[i][j]) - std::get<1>(clusters[i][l]), 2);
          a += dist;
        }
      }
      a = a / (clusters[i].size() - 1); // Average distance within cluster i

      for(int l = 0; l < k; l++) {
        if(l == i) continue;
        for(int m = 0; m < clusters[l].size(); m++) {
          float dist = pow(std::get<0>(clusters[i][j]) - std::get<0>(clusters[l][m]), 2) + pow(std::get<1>(clusters[i][j]) - std::get<1>(clusters[l][m]), 2);
          b = std::min(b, dist); // Find the smallest distance to other clusters
        }
      }

      float s; // Silhouette value for the data point
      if(clusters[i].size() == 1) {
        s = 0;
      } else {
        s = (b - a) / std::max(a, b);
      }

      silhouette += s;
    }
  }

  silhouette /= match.size(); // Average silhouette value across all data points
  return silhouette;
}

std::vector<std::vector<std::tuple<int, int, int>>> k_means(std::vector<std::tuple<int, int, int>> match, int k) {
  
  // Initialize the first centroid comptuing the average of all the points and choosing the farthest one
  std::vector<std::tuple<int, int, int>> centroids;
  int R_average = 0, G_average = 0, B_average = 0;
  for(int i = 0; i < match.size(); i++) {
    R_average += std::get<0>(match[i]);
    G_average += std::get<1>(match[i]);
    B_average += std::get<2>(match[i]);
  }
  R_average = R_average / match.size();
  G_average = G_average / match.size();
  B_average = B_average / match.size();
  
  // Find the farthest point from the average
  int max_distance = 0;
  std::tuple<int, int, int> max_centroid;
  for(int i = 0; i < match.size(); i++) {
    int distance = sqrt(pow(std::get<0>(match[i]) - R_average, 2) + pow(std::get<1>(match[i]) - G_average, 2) + pow(std::get<2>(match[i]) - B_average, 2));
    if(distance > max_distance) {
      max_distance = distance;
      max_centroid = match[i];
    }
  }
  centroids.push_back(max_centroid);

  // Initialize the other centroids
  for(int i = 1; i < k; i++) {
    int max_distance = 0;
    std::tuple<int, int, int> max_centroid;
    for(int j = 0; j < match.size(); j++) {
      int distance = 0;
      for(int l = 0; l < centroids.size(); l++) {
        distance += sqrt(pow(std::get<0>(match[j]) - std::get<0>(centroids[l]), 2) + pow(std::get<1>(match[j]) - std::get<1>(centroids[l]), 2) + pow(std::get<2>(match[j]) - std::get<2>(centroids[l]), 2));
      }
      if(distance > max_distance) {
        max_distance = distance;
        max_centroid = match[j];
      }
    }
    centroids.push_back(max_centroid);
  }

  // Initialize the clusters
  std::vector<std::vector<std::tuple<int, int, int>>> clusters;
  for(int i = 0; i < k; i++) {
    std::vector<std::tuple<int, int, int>> cluster;
    clusters.push_back(cluster);
  }

  // Assign each point to the nearest centroid
  for(int i = 0; i < match.size(); i++) {
    int min_distance = 1000000;
    int min_centroid;
    for(int j = 0; j < centroids.size(); j++) {
      int distance = sqrt(pow(std::get<0>(match[i]) - std::get<0>(centroids[j]), 2) + pow(std::get<1>(match[i]) - std::get<1>(centroids[j]), 2) + pow(std::get<2>(match[i]) - std::get<2>(centroids[j]), 2));
      if(distance < min_distance) {
        min_distance = distance;
        min_centroid = j;
      }
    }
    clusters[min_centroid].push_back(match[i]);
    // Update the centroid
    int R_average = 0, G_average = 0, B_average = 0;
    for(int j = 0; j < clusters[min_centroid].size(); j++) {
      R_average += std::get<0>(clusters[min_centroid][j]);
      G_average += std::get<1>(clusters[min_centroid][j]);
      B_average += std::get<2>(clusters[min_centroid][j]);
    }
    R_average = R_average / clusters[min_centroid].size();
    G_average = G_average / clusters[min_centroid].size();
    B_average = B_average / clusters[min_centroid].size();
    centroids[min_centroid] = std::make_tuple(R_average, G_average, B_average);
  }

  return clusters;

}

std::vector<std::vector<std::tuple<int,int>>> k_means(std::vector<std::tuple<int,int>> match, int k) {

  // Initialize the first centroid comptuing the average of all the points and choosing the farthest one
  std::vector<std::tuple<int, int>> centroids;
  int average_1 = 0;
  int average_2 = 0;
  for(int i = 0; i < match.size(); i++) {
    average_1 += std::get<0>(match[i]);
    average_2 += std::get<1>(match[i]);
  }
  average_1 = average_1 / match.size();
  average_2 = average_2 / match.size();

  // Find the farthest point from the average
  int max_distance = 0;
  std::tuple<int, int> max_centroid;
  for(int i = 0; i < match.size(); i++) {
    int distance = sqrt(pow(std::get<0>(match[i]) - average_1, 2) + pow(std::get<1>(match[i]) - average_2, 2));
    if(distance > max_distance) {
      max_distance = distance;
      max_centroid = match[i];
    }
  }
  centroids.push_back(max_centroid);

  // Initialize the other centroids
  for(int i = 1; i < k; i++) {
    int max_distance = 0;
    std::tuple<int, int> max_centroid;
    for(int j = 0; j < match.size(); j++) {
      int distance = 0;
      for(int l = 0; l < centroids.size(); l++) {
        distance += sqrt(pow(std::get<0>(match[j]) - std::get<0>(centroids[l]), 2) + pow(std::get<1>(match[j]) - std::get<1>(centroids[l]), 2));
      }
      if(distance > max_distance) {
        max_distance = distance;
        max_centroid = match[j];
      }
    }
    centroids.push_back(max_centroid);
  }

  // Initialize the clusters
  std::vector<std::vector<std::tuple<int, int>>> clusters;
  for(int i = 0; i < k; i++) {
    std::vector<std::tuple<int, int>> cluster;
    clusters.push_back(cluster);
  }

  // Assign each point to the nearest centroid
  for(int i = 0; i < match.size(); i++) {
    int min_distance = 1000000;
    int min_centroid;
    for(int j = 0; j < centroids.size(); j++) {
      int distance = sqrt(pow(std::get<0>(match[i]) - std::get<0>(centroids[j]), 2) + pow(std::get<1>(match[i]) - std::get<1>(centroids[j]), 2));
      if(distance < min_distance) {
        min_distance = distance;
        min_centroid = j;
      }
    }
    clusters[min_centroid].push_back(match[i]);
    // Update the centroid
    int average_1 = 0, average_2 = 0;
    for(int j = 0; j < clusters[min_centroid].size(); j++) {
      average_1 += std::get<0>(clusters[min_centroid][j]);
      average_2 += std::get<1>(clusters[min_centroid][j]);
    }
    average_1 = average_1 / clusters[min_centroid].size();
    average_2 = average_2 / clusters[min_centroid].size();
    centroids[min_centroid] = std::make_tuple(average_1, average_2);
  }

  return clusters;  

}

float findMax(float a, float b, float c) {
  float maxVal = a;

  if(b > maxVal) {
    maxVal = b;
  }

  if(c > maxVal) {
    maxVal = c;
  }

  return maxVal;

}

void localizePlayers(const cv::Mat& original_image, const cv::Mat& mask, std::vector<std::tuple<cv::Rect, cv::Mat, int>>& players) {
  for(int i = 1; i < 256; i++) {
    cv::Mat player_mask = mask.clone();
    cv::Mat original_image_copy = original_image.clone();
    bool found = false;
    for(int j = 0; j < original_image.rows; j++) {
      for(int k = 0; k < original_image.cols; k++) {
        if(player_mask.at<uchar>(j, k) == i) {
          found = true;
        }
        if(player_mask.at<uchar>(j, k) != i) {
          player_mask.at<uchar>(j, k) = 0;
        }
      }
    }
    if(found) {

      // Select the area with color i and extract the bounding box
      cv::Rect bounding_box = cv::boundingRect(player_mask);

      // Draw the bounding box on the original image
      cv::rectangle(original_image_copy, bounding_box, cv::Scalar(0, 255, 0), 2);
      cv::Mat player_bounding_box = original_image_copy(bounding_box);

      // Apply gaussian blur
      cv::GaussianBlur(player_bounding_box, player_bounding_box, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
      cv::GaussianBlur(player_bounding_box, player_bounding_box, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
      cv::GaussianBlur(player_bounding_box, player_bounding_box, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
      cv::GaussianBlur(player_bounding_box, player_bounding_box, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
      cv::GaussianBlur(player_bounding_box, player_bounding_box, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

      // Apply pyramid mean shift filtering
      cv::pyrMeanShiftFiltering(player_bounding_box, player_bounding_box, 10, 20, 2);

      // Color black all the pixels that are not part of the player
      for(int j = 0; j < player_bounding_box.rows; j++) {
        for(int k = 0; k < player_bounding_box.cols; k++) {
          if(player_mask.at<uchar>(j + bounding_box.y, k + bounding_box.x) != i) {
            player_bounding_box.at<cv::Vec3b>(j, k)[0] = 0;
            player_bounding_box.at<cv::Vec3b>(j, k)[1] = 0;
            player_bounding_box.at<cv::Vec3b>(j, k)[2] = 0;
          }
        }
      }

      players.push_back(std::make_tuple(bounding_box, player_bounding_box, i));
    }

  }
}

void parseClusters(std::vector<std::vector<std::tuple<int, int, int>>> clusters, std::vector<std::tuple<int, int, int>> match, std::vector<int>& team_membership) {
  for(int i = 0; i < clusters.size(); i++) {
    for(int j = 0; j < clusters[i].size(); j++) {
      // find the index of the point in the match vector
      for(int k = 0; k < match.size(); k++) {
        if(std::get<0>(clusters[i][j]) == std::get<0>(match[k]) && std::get<1>(clusters[i][j]) == std::get<1>(match[k]) && std::get<2>(clusters[i][j]) == std::get<2>(match[k])) {
          //std::cout << "Player " << k + 1 << " belongs to cluster " << i+1 << std::endl;
          team_membership[k] = i;
        }
      }
    }
  }
}

void parseClusters(std::vector<std::vector<std::tuple<int, int>>> clusters, std::vector<std::tuple<int, int>> match, std::vector<int>& team_membership) {
  for(int i = 0; i < clusters.size(); i++) {
    for(int j = 0; j < clusters[i].size(); j++) {
      // find the index of the point in the match vector
      for(int k = 0; k < match.size(); k++) {
        if(std::get<0>(clusters[i][j]) == std::get<0>(match[k]) && std::get<1>(clusters[i][j]) == std::get<1>(match[k])) {
          //std::cout << "Player " << k + 1 << " belongs to cluster " << i+1 << std::endl;
          team_membership[k] = i;
        }
      }
    }
  }
}

void saveOutput(const std::string& output_folder_path, const std::string& file_name, const cv::Mat& RGB_mask, const cv::Mat& BN_mask, const std::vector<std::tuple<cv::Rect, cv::Mat, int>>& players, const std::vector<int>& team_membership) {

  // Check if the output folder exists
  if(!cv::utils::fs::exists(output_folder_path)) {
    printf("The folder %s does not exist\n", output_folder_path.c_str());
    printf("Creating the folder %s\n", output_folder_path.c_str());
    cv::utils::fs::createDirectories(output_folder_path);
  }

  // Check if the output folder contains the Masks folder
  if(!cv::utils::fs::exists(output_folder_path + "/Masks")) {
    cv::utils::fs::createDirectories(output_folder_path + "/Masks");
  }

  std::string img_name = file_name.substr(file_name.find_last_of("/") + 1);
  img_name.erase(img_name.length() - 4);

  // Save the RGB mask
  std::string RGB_mask_path = output_folder_path + "/Masks/" + img_name + "_color.png";
  cv::imwrite(RGB_mask_path, RGB_mask);

  // Save the BN mask
  std::string BN_mask_path = output_folder_path + "/Masks/" + img_name + "_bin.png";
  cv::imwrite(BN_mask_path, BN_mask);

  // Save the bounding boxes of the players
  std::string coordinates_path = output_folder_path + "/Masks/" + img_name + "_bb.txt";
  std::ofstream coordinates_file(coordinates_path);
  for(int i = 0; i < players.size(); i++) {
    coordinates_file << std::get<0>(players[i]).x << " " << std::get<0>(players[i]).y << " " << std::get<0>(players[i]).width << " " << std::get<0>(players[i]).height << " " << team_membership[i] + 1 << std::endl;
  }
  coordinates_file.close();

}

void assignToTeams(const std::string& output_folder_path, std::string file_name, cv::Mat& original_image, cv::Mat& mask, cv::Mat& field_mask) {
  
  std::vector<std::tuple<cv::Rect, cv::Mat, int>> players; // Vector containing the Rect object of the bounding box of the player, the Mat object containing the image of the player and the colorID of the player in the mask
  std::vector<std::tuple<int, int, int>> match;
  cv::Mat RGB_mask = cv::Mat::zeros(mask.size(), CV_8UC3);
  cv::Mat BN_mask = cv::Mat::zeros(mask.size(), CV_8UC1);

  // Copy the field_mask in the BN_mask
  for(int i = 0; i < field_mask.rows; i++) {
    for(int j = 0; j < field_mask.cols; j++) {
      BN_mask.at<uchar>(i, j) = field_mask.at<uchar>(i, j);
    }
  }
  // Copy the field_mask in the RGB_mask
  for(int i = 0; i < field_mask.rows; i++) {
    for(int j = 0; j < field_mask.cols; j++) {
      if(field_mask.at<uchar>(i, j) == 3) {
        RGB_mask.at<cv::Vec3b>(i, j)[0] = 0;
        RGB_mask.at<cv::Vec3b>(i, j)[1] = 255;
        RGB_mask.at<cv::Vec3b>(i, j)[2] = 0;
      }
    }
  }

  localizePlayers(original_image, mask, players);
  
  std::vector<int> team_membership(players.size(), -1);

  for(int i = 0; i < players.size(); i++) {
  
    // Compute average of the three channels
    std::vector<int> R, G, B;
    for(int k = 0; k < std::get<1>(players[i]).rows; k++) {
      for(int l = 0; l < std::get<1>(players[i]).cols; l++) {
        cv::Vec3b pixel = std::get<1>(players[i]).at<cv::Vec3b>(k, l);
        int b = pixel[0];
        int g = pixel[1];
        int r = pixel[2];
        if(r < 10 || g < 10 || b < 10) //TODO controlla queste threshold
          continue;
        R.push_back(r);
        G.push_back(g);
        B.push_back(b);
        int average = (r + g + b) / 3;
      }
    }

    // Compute the average of the three channels
    int R_average = 0, G_average = 0, B_average = 0;
    for(int k = 0; k < R.size(); k++) {
      R_average += R[k];
      G_average += G[k];
      B_average += B[k];
    }

    R_average = R_average / R.size();
    G_average = G_average / G.size();
    B_average = B_average / B.size();

    match.push_back(std::make_tuple(R_average, G_average, B_average));

  }

  // Compute the mean distance between all the points
  float distance = 0;
  for(int k = 0; k < match.size(); k++) {
    for(int l = k + 1; l < match.size(); l++) {
      distance += sqrt(pow(std::get<0>(match[k]) - std::get<0>(match[l]), 2) + pow(std::get<1>(match[k]) - std::get<1>(match[l]), 2) + pow(std::get<2>(match[k]) - std::get<2>(match[l]), 2));
  //std::cout << "Distance: " << distance << std::endl;
    }
  }
  distance = distance / (match.size() * (match.size() - 1) / 2);

  if(distance > 0) {
    // Create a matchGB vector
    std::vector<std::tuple<int, int>> matchGB;
    for(int k = 0; k < match.size(); k++) {
      int G = std::get<1>(match[k]);
      int B = std::get<2>(match[k]);
      matchGB.push_back(std::make_tuple(G, B));
    }
    std::vector<std::tuple<int, int>> matchRG;
    for(int k = 0; k < match.size(); k++) {
      int R = std::get<0>(match[k]);
      int G = std::get<1>(match[k]);
      matchRG.push_back(std::make_tuple(R, G));
    }

    std::vector<std::vector<std::tuple<int, int, int>>> clusters3dk2 = k_means(match, 2);
    std::vector<std::vector<std::tuple<int, int>>> clustersGBk2 = k_means(matchGB, 2);
    std::vector<std::vector<std::tuple<int, int>>> clustersGRk2 = k_means(matchRG, 2);

    float sil3dk2 = silhouette(match, clusters3dk2, 2);
    float silGBk2 = silhouette(matchGB, clustersGBk2, 2);
    float silRGk2 = silhouette(matchRG, clustersGRk2, 2);

    // Print the min silhouette value
    std::cout << "Silhouette 3d k=2: " << sil3dk2 << std::endl;
    std::cout << "Silhouette GB k=2: " << silGBk2 << std::endl;
    std::cout << "Silhouette RG k=2: " << silRGk2 << std::endl;

    float max_silhouette = findMax(sil3dk2, silGBk2, silRGk2);

    if(max_silhouette == sil3dk2) {
      std::cout << "3d k=2" << std::endl;
      parseClusters(clusters3dk2, match, team_membership);
    } else if(max_silhouette == silGBk2) {
      std::cout << "GB k=2" << std::endl;
      parseClusters(clustersGBk2, matchGB, team_membership);
    } else if(max_silhouette == silRGk2) {
      std::cout << "RG k=2" << std::endl;
      parseClusters(clustersGRk2, matchRG, team_membership);
    }

  }

  // For each player we color his mask with the color of his team based on the team_membership vector (0 is blue, 1 is red, 2 is yellow)
  for(int i = 0; i < players.size(); i++) {
    if(team_membership[i] == 0) {
      // Color the mask of the player with blue
      for(int j = 0; j < mask.rows; j++) {
        for(int k = 0; k < mask.cols; k++) {
          if(mask.at<uchar>(j, k) == std::get<2>(players[i])) {
            RGB_mask.at<cv::Vec3b>(j, k) = cv::Vec3b(255, 0, 0);
            BN_mask.at<uchar>(j, k) = 1;
          }
        }
      }
    } else if(team_membership[i] == 1) {
      // Color the mask of the player with red
      for(int j = 0; j < mask.rows; j++) {
        for(int k = 0; k < mask.cols; k++) {
          if(mask.at<uchar>(j, k) == std::get<2>(players[i])) {
            RGB_mask.at<cv::Vec3b>(j, k) = cv::Vec3b(0, 0, 255);
            BN_mask.at<uchar>(j, k) = 2;
          }
        }
      }
    }
  }

  saveOutput(output_folder_path, file_name, RGB_mask, BN_mask, players, team_membership);

}
