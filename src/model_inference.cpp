#include <opencv2/opencv.hpp>
#include "postProcessing.h"
#include "playerAssignement.h"

/**
 * This main can be used on the entire dataset to generate the output files:
 *
 * Run the following command:
 * ./model_inference <path_to_model> <path_to_folder> <path_to_output_folder>
 * 
 */
int main(int argc, char **argv)
{
  if (argc < 3)
  {
    printf("Usage 1: ./model_inference <path_to_model> <path_to_image>\n");
    printf("Usage 2: ./model_inference <path_to_model> <path_to_folder> <path_to_output_folder>\n");
    return -1;
  }

  if (argc == 4)
  {
    printf("Running the algorithm on all images inside folder...\n");
    string model_path = argv[1];
    string folder_path = argv[2];
    string output_folder_path = argv[3];
    playerAssignement(model_path, folder_path, output_folder_path);
    // TODO funzione di Federico
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