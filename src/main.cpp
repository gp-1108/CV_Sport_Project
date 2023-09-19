#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "yolo.h"
#include "sceneAnalyzer.h"
#include "performances.h"

/**
 * This main can be used on the entire dataset to generate the output files:
 * 
 * Run the following command:
 * ./model_inference <path_to_model> <path_to_folder> <path_to_output_folder>
 * 
 */
int main(int argc, char **argv)
{
  if (argc == 4)
  {
    printf("Running the algorithm on all images inside folder...\n");
    std::string model_path = argv[1];
    std::string folder_path = argv[2];
    std::string output_folder_path = argv[3];
    
    // Model initialization
    Yolov8Seg yolo(model_path);

    // Checking if the folders exist
    if (!cv::utils::fs::exists(folder_path)) {
      printf("The folder %s does not exist\n", folder_path.c_str());
      return -1;
    }

    std::string src_images_path = folder_path + "/Images";
    // Checking if the folders exist
    if (!cv::utils::fs::exists(src_images_path)) {
      printf("The folder %s does not exist\n", src_images_path.c_str());
      return -1;
    }

    std::vector<std::string> file_names; // The names of the files in the folder
    cv::glob(src_images_path, file_names); // Getting the names of the files in the folder

    std::printf("Detected %ld files in the folder %s\n", file_names.size(), folder_path.c_str());

    for (int i = 0; i < file_names.size(); i++) {
      std::cout << "Processing file " << file_names[i] << std::endl;

      // Checking if the file is an image
      if (file_names[i].find(".jpg") == std::string::npos && file_names[i].find(".png") == std::string::npos) {
        continue;
      }

      sceneAnalyzer(yolo, output_folder_path, file_names[i]);

    }

    std::printf("Processing done!\n");

    std::printf("#############################################\n\n\n");
    std::printf("Now starting with performance evaluation...\n");

    // Performance evaluation
    std::string ground_truth_path = folder_path + "/Masks";
    std::string output_pred_path = output_folder_path + "/Masks";

    computeMetrics(output_pred_path, ground_truth_path);

    // Remove the confidences from the txt file
    for (int i = 0; i < file_names.size(); i++) {
      txtCleanUp(output_folder_path, file_names[i]);
    }

    std::printf("Performance evaluation done!\n\n");

    std::printf("Exiting the program...\n");
    return 0;
  }
  else
  {
    printf("Invalid number of arguments!\n");
    printf("Usage: ./model_inference <path_to_model> <path_to_folder> <path_to_output_folder>\n");
    return -1;
  }
}