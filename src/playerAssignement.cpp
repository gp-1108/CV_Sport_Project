/**
* @file playerAssignement.cpp
* @author Enrico D'Alberton ID number: 2093708
* @date ---
* @version 1.0
*/
#include "playerAssignement.h"
#include <opencv2/opencv.hpp>
#include <tuple>
#include <algorithm>
#include <vector>

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

float findMax(float a, float b, float c, float d, float e, float f) {
  float maxVal = a;

  if(b > maxVal) {
    maxVal = b;
  }

  if(c > maxVal) {
    maxVal = c;
  }

  if(d > maxVal) {
    maxVal = d;
  }

  if(e > maxVal) {
    maxVal = e;
  }

  if(f > maxVal) {
    maxVal = f;
  }

  return maxVal;

}

void localizePlayers(const cv::Mat& original_image, const cv::Mat& mask, std::vector<std::tuple<cv::Rect, cv::Mat, int>>& players) {
  for(int i = 1; i < 256; i++) {
    cv::Mat player_mask = mask.clone();
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
      // Find the only bounding box of the player
      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(player_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      cv::Rect bounding_box = cv::boundingRect(contours[0]);
      cv::Mat player_bounding_box = original_image(bounding_box);
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
          if(player_mask.at<uchar>(j, k) != i) {
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

void parseClusters(std::vector<std::vector<std::tuple<int, int, int>>> clusters, std::vector<std::tuple<int, int, int>> match, std::vector<int> team_membership) {
  for(int i = 0; i < clusters.size(); i++) {
    for(int j = 0; j < clusters[i].size(); j++) {
      // find the index of the point in the match vector
      for(int k = 0; k < match.size(); k++) {
        if(std::get<0>(clusters[i][j]) == std::get<0>(match[k]) && std::get<1>(clusters[i][j]) == std::get<1>(match[k]) && std::get<2>(clusters[i][j]) == std::get<2>(match[k])) {
          std::cout << "Player " << k + 1 << " belongs to cluster " << i+1 << std::endl;
          team_membership[k] = i;
        }
      }
    }
  }
}

void parseClusters(std::vector<std::vector<std::tuple<int, int>>> clusters, std::vector<std::tuple<int, int>> match, std::vector<int> team_membership) {
  for(int i = 0; i < clusters.size(); i++) {
    for(int j = 0; j < clusters[i].size(); j++) {
      // find the index of the point in the match vector
      for(int k = 0; k < match.size(); k++) {
        if(std::get<0>(clusters[i][j]) == std::get<0>(match[k]) && std::get<1>(clusters[i][j]) == std::get<1>(match[k])) {
          std::cout << "Player " << k + 1 << " belongs to cluster " << i+1 << std::endl;
          team_membership[k] = i;
        }
      }
    }
  }
}


void assignToTeams(const std::string& output_folder_path, cv::Mat& original_image, cv::Mat& mask) {
  
  std::vector<std::tuple<cv::Rect, cv::Mat, int>> players; // Vector containing the Rect object of the bounding box of the player, the Mat object containing the image of the player and the colorID of the player in the mask
  std::vector<int> team_membership(players.size(), -1); 
  std::vector<std::tuple<int, int, int>> match;

  localizePlayers(original_image, mask, players);

  for(int i = 0; i < players.size(); i++) {
  
    // Compute average of the three channels
    std::vector<int> R, G, B;
    for(int k = 0; k < image.rows; k++) {
      for(int l = 0; l < image.cols; l++) {
        cv::Vec3b pixel = image.at<cv::Vec3b>(k, l);
        int b = pixel[0];
        int g = pixel[1];
        int r = pixel[2];
        if(r < 10 || g < 10 || b < 10) //TODO controlla quesste threshold
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
    }
  }
  distance = distance / (match.size() * (match.size() - 1) / 2);
  std::cout << "Mean distance: " << distance << std::endl;
  mean_distance.push_back(distance);

  if(distance > 0) {
    std::cout << "K-Means" << std::endl;
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
    std::vector<std::vector<std::tuple<int, int, int>>> clusters3dk3 = k_means(match, 3);
    std::vector<std::vector<std::tuple<int, int>>> clustersGBk2 = k_means(matchGB, 2);
    std::vector<std::vector<std::tuple<int, int>>> clustersGBk3 = k_means(matchGB, 3);
    std::vector<std::vector<std::tuple<int, int>>> clustersGRk2 = k_means(matchRG, 2);
    std::vector<std::vector<std::tuple<int, int>>> clustersGRk3 = k_means(matchRG, 3);

    float sil3dk2 = silhouette(match, clusters3dk2, 2);
    float sil3dk3 = silhouette(match, clusters3dk3, 3);
    float silGBk2 = silhouette(matchGB, clustersGBk2, 2);
    float silGBk3 = silhouette(matchGB, clustersGBk3, 3);
    float silRGk2 = silhouette(matchRG, clustersGRk2, 2);
    float silRGk3 = silhouette(matchRG, clustersGRk3, 3);

    // Print the min silhouette value
    std::cout << "Silhouette 3d k=2: " << sil3dk2 << std::endl;
    std::cout << "Silhouette 3d k=3: " << sil3dk3 << std::endl;
    std::cout << "Silhouette GB k=2: " << silGBk2 << std::endl;
    std::cout << "Silhouette GB k=3: " << silGBk3 << std::endl;
    std::cout << "Silhouette RG k=2: " << silRGk2 << std::endl;
    std::cout << "Silhouette RG k=3: " << silRGk3 << std::endl;
  
    if(findMax(sil3dk2, sil3dk3, silGBk2, silGBk3, silRGk2, silRGk3) == sil3dk2) {
      std::cout << "3d k=2" << std::endl;
      parseClusters(clusters3dk2, match, team_membership);
    } else if(findMax(sil3dk2, sil3dk3, silGBk2, silGBk3, silRGk2, silRGk3) == sil3dk3) {
      std::cout << "3d k=3" << std::endl;
      parseClusters(clusters3dk3, match, team_membership);
    } else if(findMax(sil3dk2, sil3dk3, silGBk2, silGBk3, silRGk2, silRGk3) == silGBk2) {
      std::cout << "GB k=2" << std::endl;
      parseClusters(clustersGBk2, matchGB, team_membership);
    } else if(findMax(sil3dk2, sil3dk3, silGBk2, silGBk3, silRGk2, silRGk3) == silGBk3) {
      std::cout << "GB k=3" << std::endl;
      parseClusters(clustersGBk3, matchGB, team_membership);
    } else if(findMax(sil3dk2, sil3dk3, silGBk2, silGBk3, silRGk2, silRGk3) == silRGk2) {
      std::cout << "RG k=2" << std::endl;
      parseClusters(clustersGRk2, matchRG, team_membership);
    } else if(findMax(sil3dk2, sil3dk3, silGBk2, silGBk3, silRGk2, silRGk3) == silRGk3) {
      std::cout << "RG k=3" << std::endl;
      parseClusters(clustersGRk3, matchRG, team_membership);
    }

  }


}

void playerAssignement(const Yolov8Seg& yolo, const std::string& folder_path, const std::string& output_folder_path) {
  
  // Checking if the folders exist
  if (!cv::utils::fs::exists(folder_path)) {
    printf("The folder %s does not exist\n", folder_path.c_str());
    return;
  }
  if (!cv::utils::fs::exists(output_folder_path)) {
    printf("The folder %s does not exist\n", output_folder_path.c_str());
    printf("Creating the folder %s\n", output_folder_path.c_str());
    cv::utils::fs::createDirectories(output_folder_path);
  }

  std::vector<cv::String> file_names; // The names of the files in the folder
  cv::glob(folder_path, file_names); // Getting the names of the files in the folder

  std::printf("Detected %ld files in the folder %s\n", file_names.size(), folder_path.c_str());

  for (int i = 0; i < file_names.size(); i++) {
    std::printf("Processing file %d/%ld\n", i + 1, file_names.size());

    // Checking if the file is an image
    if (file_names[i].find(".jpg") == std::string::npos && file_names[i].find(".png") == std::string::npos) {
      continue;
    }
    // Reading the image
    cv::Mat originalImage = cv::imread(file_names[i]);

    // Running the segmentation

    cv::Mat finalMask;
    yolo.runSegmentation(originalImage, finalMask);   
    finalMask = postProcessing(originalImage, finalMask); //TODO modifica direttamente la reference
    assignToTeams(output_folder_path, originalImage, finalMask);
  }
}