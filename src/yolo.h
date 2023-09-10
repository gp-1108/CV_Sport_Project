#ifndef YOLONET_H
#define YOLONET_H

#include <opencv2/dnn.hpp>
#include <vector>

class Yolov8Seg {
  public:
    Yolov8Seg(std::string model_path);
    // [TODO] Add other constructors for different thresholds
    void runSegmentation(const cv::Mat& src_img, cv::Mat& final_mask);
    ~Yolov8Seg();
    cv::Mat displaySegOutput(const cv::Mat& src_img, const cv::Mat& final_mask);


  private:
    struct OutputSeg {
      cv::Rect box;
      int class_id;
      float confidence;
      std::vector<float> mask;
      cv::Mat box_mask;
    };

    void preProcessImage(const cv::Mat& src_img, cv::Mat& net_input_img, cv::Vec4d& conv_params);

    void generateBoxMask(const cv::Mat& mask_candidate, const cv::Mat& mask_protos, OutputSeg &output, cv::Vec4d& conv_params, const cv::Size& src_img_shape);

    cv::Size net_size = cv::Size(640, 640); // The net image size
    cv::dnn::Net net; // The neural network
    std::vector<std::string> classes {"person"}; // The classes that the model can detect
    float class_threshold = 0.25;
    float nms_threshold = 0.45;
    float mask_threshold = 0.5;

    int seg_channels = 32;
    int seg_width = 160;
    int seg_height = 160;

};

#endif // YOLONET_H