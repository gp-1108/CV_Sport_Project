#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "yolo.h"
#include "sceneAnalyzer.h"

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
    if (!cv::utils::fs::exists(output_folder_path)) {
      printf("The folder %s does not exist\n", output_folder_path.c_str());
      printf("Creating the folder %s\n", output_folder_path.c_str());
      cv::utils::fs::createDirectories(output_folder_path);
    }

    std::vector<std::string> file_names; // The names of the files in the folder
    cv::glob(folder_path, file_names); // Getting the names of the files in the folder

    std::printf("Detected %ld files in the folder %s\n", file_names.size(), folder_path.c_str());

    for (int i = 0; i < file_names.size(); i++) {
      std::cout << "Processing file " << file_names[i] << std::endl;

      // Checking if the file is an image
      if (file_names[i].find(".jpg") == std::string::npos && file_names[i].find(".png") == std::string::npos) {
        continue;
      }
      // Reading the image
      cv::Mat original_image = cv::imread(file_names[i]);

      sceneAnalyzer(yolo, output_folder_path, file_names[i]);

      // Running the segmentation
      //cv::Mat finalMask;
      //yolo.runSegmentation(original_image, finalMask);
      //finalMask = postProcessing(originalImage, finalMask); //TODO modifica direttamente la reference
      //assignToTeams(output_folder_path, file_names[i], originalImage, finalMask);
    }

    printf("Done!\n");
    return 0;
  }
  else
  {
    printf("Invalid number of arguments!\n");
    printf("Usage: ./model_inference <path_to_model> <path_to_folder> <path_to_output_folder>\n");
    return -1;
  }
}