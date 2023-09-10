#include "yolo.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
  string model_path = argv[1];
  Yolov8Seg yolo(model_path);

  string img_path = argv[2];
  Mat src_img = imread(img_path);

  Mat final_mask;
  yolo.runSegmentation(src_img, final_mask);

  cv::Mat display_img = yolo.displaySegOutput(src_img, final_mask);

  src_img = 0.5 * src_img + 0.5 * display_img;
  imshow("src_img", src_img);
  waitKey(0);
  destroyAllWindows();

  return 0;
}