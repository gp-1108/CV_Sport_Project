#ifndef YOLONET_H
#define YOLONET_H

#include <opencv2/dnn.hpp>
#include <vector>

class Yolov8Seg {
  public:
    /**
     * @brief Construct a new Yolov8Seg object
     * 
     * Initialize a neural network with the given model file.
     * It must be a YOLOv8 model trained for segmentation with a single class.
     * The defualt constructor uses the following parameters:
     * - net_size = cv::Size(640, 640)
     * - classes = {"person"}
     * - class_threshold = 0.25
     * - nms_threshold = 0.45
     * - mask_threshold = 0.5
     * - seg_channels = 32
     * - seg_width = 160
     * - seg_height = 160
     * The model is meant to be run on CPU.
     * 
     * @param model_path The path to the model file
    */
    Yolov8Seg(std::string model_path);

    /**
     * @brief Destroy the Yolov8Seg object
    */
    ~Yolov8Seg();
    // [TODO] Add other constructors for different thresholds

    /**
     * @brief Run the segmentation on the given image
     * 
     * The function takes as an argument the source image in BGR format
     * and assigns to the final_mask parameter the final gray scale mask of the detected object.
     * Each pixel of the mask is a value between 0 and 255. 
     * 0 means that the pixel is not of interest
     * any other value > 0 is a label for segmentation.
     * An object is composed of a set of pixels with the same label.
     * 
     * @param src_img The source image
     * @param final_mask The final mask
    */
    void runSegmentation(
      const cv::Mat& src_img, // The source image
      cv::Mat& final_mask // Where to save the final mask
    );

    /**
     * @brief Display the output of the segmentation
     * 
     * The function takes as an argument the source image in BGR format
     * and the final mask and returns a new image with the mask applied to the source image.
     * The final mask is the output of the Yolov8Seg::runSegmentation() function.
     * It returns a new image with the colored version of the gray scale mask
     * for visualization purposes.
     * Each object is colored with a different random color.
     * 
     * @param src_img The source image
     * @param final_mask The final mask in the Yolov8Seg::runSegmentation() function
     * @return cv::Mat The colored mask
    */
    cv::Mat displaySegOutput(
      const cv::Mat& src_img, // The source image
      const cv::Mat& final_mask // The final mask on the Yolov8Seg::runSegmentation() function
    );
  
    /**
     * @brief Run the segmentation on the given folder of images
     * 
     * The function takes as an argument the path to the folder containing the images
     * and the path to the folder where to save the output images.
     * The output images are saved in the output folder with the same name as the input images.
     * Running the inference on each image and saving the output of the displaySegOutput() function.
     * 
     * @param folder_path The path to the folder containing the images
     * @param output_folder_path The path to the folder where to save the output images
    */
    void runOnFolder(
      const std::string& folder_path, // The path to the folder
      const std::string& output_folder_path // The path to the output folder
    );


  private:
    struct OutputSeg {
      cv::Rect box;
      int class_id;
      float confidence;
      std::vector<float> mask;
      cv::Mat box_mask;
    }; // A struct to store the output of the segmentation for each detected object

    /**
     * @brief Preprocess the image for the neural network
     * 
     * This function takes as an argument the source image in BGR format
     * Pads the image to the net size with gray pixels and saves the conversion parameters
     * in the conv_params parameter.
     * 
     * @param src_img The source image
     * @param net_input_img The preprocessed image
     * @param conv_params The conversion parameters. [width_ratio, height_ratio, pad_width, pad_height]
    */
    void preProcessImage(
      const cv::Mat& src_img, // The source image
      cv::Mat& net_input_img, // Where to save the preprocessed image
      cv::Vec4d& conv_params // Where to save the conversion parameters
    );

    /**
     * @brief Generate the mask for the detected object given the segmentation output
     * 
     * This function takes as an argument the segmentation output for a single object
     * and generates the mask for the object with respect to its bounding box.
     * As such it is not a standalone function and it is used in the Yolov8Seg::runSegmentation() function.
     * The mask is saved in the output.box_mask parameter.
     * 
     * @param mask_candidate The segmentation output for a single object
     * @param mask_protos The segmentation output for a single object
     * @param output The output of the segmentation for a single object
     * @param conv_params The conversion parameters. [width_ratio, height_ratio, pad_width, pad_height]
     * @param src_img_shape The shape of the source image
    */
    void generateBoxMask(
      const cv::Mat& mask_candidate, // The segmentation output for a single object
      const cv::Mat& mask_protos, // The network output for the mask prototypes
      OutputSeg &output, // The output of the segmentation for a single object
      cv::Vec4d& conv_params, // The conversion parameters. [width_ratio, height_ratio, pad_width, pad_height]
      const cv::Size& src_img_shape // The shape of the source image
    );

    cv::Size net_size = cv::Size(640, 640); // The net image size
    cv::dnn::Net net; // The neural network
    std::vector<std::string> classes {"person"}; // The classes that the model can detect
    float class_threshold = 0.25; // The threshold for the confidence of the detected object
    float nms_threshold = 0.45; // The threshold for the non-maximum suppression
    float mask_threshold = 0.5; // The threshold for the mask of the detected object

    int seg_channels = 32; // The number of channels of the mask
    int seg_width = 160;  // The width of the mask
    int seg_height = 160; // The height of the mask
};

#endif // YOLONET_H