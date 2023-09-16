#include "yolo.h"
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;

/**
 * This main can be used in two ways:
 * 1. To run inference on a single image
 * 2. To run inference on all images inside a folder
 *
 * For 1, run the following command:
 * ./model_inference <path_to_model> <path_to_image>
 *
 * For 2, run the following command:
 * ./model_inference <path_to_model> <path_to_folder> <path_to_output_folder>
 */
int main(int argc, char **argv)
{
  if (argc < 3)
  {
    printf("Usage 1: ./model_inference <path_to_model> <path_to_image>\n");
    printf("Usage 2: ./model_inference <path_to_model> <path_to_folder> <path_to_output_folder>\n");
    return -1;
  }

  printf("Loading model...");
  auto start = chrono::high_resolution_clock::now();
  string model_path = argv[1];
  Yolov8Seg yolo(model_path);
  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
  printf("Done! (%ld ms)\n", duration.count());

  if (argc == 4)
  {

    printf("Running inference on all images inside folder...\n");
    string folder_path = argv[2];
    string output_folder_path = argv[3];
    yolo.runOnFolder(folder_path, output_folder_path);
    printf("Done!\n");
    return 0;
  }
  else if (argc == 3)
  {

    string img_path = argv[2];
    Mat src_img = imread(img_path);

    printf("Running segmentation...\n");
    Mat final_mask;
    start = chrono::high_resolution_clock::now();
    yolo.runSegmentation(src_img, final_mask);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    printf("Done! (%ld ms)\n", duration.count());

    Mat display_img = yolo.displaySegOutput(src_img, final_mask);

    src_img = 0.5 * src_img + 0.5 * display_img;
    imshow("src_img", src_img);
    waitKey(0);
    destroyAllWindows();

    return 0;
  }
  else
  {
    printf("Invalid number of arguments!\n");
    printf("Usage 1: ./model_inference <path_to_model> <path_to_image>\n");
    printf("Usage 2: ./model_inference <path_to_model> <path_to_folder> <path_to_output_folder>\n");
    return -1;
  }
}