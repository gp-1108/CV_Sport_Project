#include "yolo.h"
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>

Yolov8Seg::~Yolov8Seg() {}

Yolov8Seg::Yolov8Seg(std::string model_path)
{
  // Checking if the model file exists
  if (!cv::utils::fs::exists(model_path))
  {
    throw std::invalid_argument("The model file does not exist");
  }
  this->net = cv::dnn::readNet(model_path); // Read the model

  // Setting the model to run on CPU
  this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
  this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void Yolov8Seg::runSegmentation(const cv::Mat &src_img, cv::Mat &final_mask)
{
  // Checking if the image is empty
  if (src_img.empty())
  {
    throw std::invalid_argument("The src image is empty, cannot run segmentation");
  }

  cv::Vec4d conv_params;                                // [ratio_w, ratio_h, pad_w, pad_h]
  cv::Mat net_input_img;                                // The preprocessed image
  preProcessImage(src_img, net_input_img, conv_params); // Preprocess the image with padding to make it fit the net size

  // Create a blob from a image with values scaled to [0..1]
  cv::Mat blob;
  cv::dnn::blobFromImage(net_input_img, blob, 1 / 255.0, this->net_size, cv::Scalar(0, 0, 0), true, false);

  // Run the model
  net.setInput(blob);
  std::vector<cv::Mat> net_output_img;
  std::vector<std::string> output_layer_names{"output0", "output1"}; // Making sure that we get the output of the segmentation in the correct order
  net.forward(net_output_img, output_layer_names);                   // get outputs

  std::vector<int> class_ids;                       // class ids
  std::vector<float> confidences;                   // object scores
  std::vector<cv::Rect> boxes;                      // bounding boxes
  std::vector<std::vector<float>> picked_proposals; // output0[:,:, 4 + _className.size():net_width] possible masks
  int net_width = this->classes.size() + 4 + this->seg_channels;

  // The output of the segmentation is a 3D matrix of shape [bs, 116, 8400]
  cv::Mat output0 = cv::Mat(cv::Size(net_output_img[0].size[2], net_output_img[0].size[1]), CV_32F, (float *)net_output_img[0].data).t(); //[bs,116,8400]=>[bs,8400,116]
  int rows = output0.rows;                                                                                                                // Number of rows in the output
  float *pdata = (float *)output0.data;                                                                                                   // Pointer used to loop through the output data

  // Loop through the output data
  for (int r = 0; r < rows; r++)
  {
    cv::Mat scores(1, this->classes.size(), CV_32FC1, pdata + 4); // get scores for each class
    cv::Point classIdPoint;

    // Findinging the class with the highest score
    double max_class_score;
    cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
    max_class_score = (float)max_class_score;

    if (max_class_score >= this->class_threshold)
    {
      std::vector<float> temp_proto(pdata + 4 + this->classes.size(), pdata + net_width); // get prototype mask
      picked_proposals.push_back(temp_proto);

      // Save it along with the bounding box and the true locations of the bounding box
      float x = (pdata[0] - conv_params[2]) / conv_params[0];
      float y = (pdata[1] - conv_params[3]) / conv_params[1];
      float w = pdata[2] / conv_params[0];
      float h = pdata[3] / conv_params[1];
      int left = MAX(int(x - 0.5 * w + 0.5), 0);
      int top = MAX(int(y - 0.5 * h + 0.5), 0);
      class_ids.push_back(classIdPoint.x);
      confidences.push_back(max_class_score);
      boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
    }

    pdata += net_width; // next line
  }

  // Non Maxima Suppresion for bounding boxes
  // to remove overlapping boxes
  std::vector<int> nms_res;                                                                   // indices of the boxes to keep
  cv::dnn::NMSBoxes(boxes, confidences, this->class_threshold, this->nms_threshold, nms_res); // Non Maxima Suppresion

  std::vector<OutputSeg> outputs; // The final output

  cv::Rect rect_bit_mask(0, 0, src_img.cols, src_img.rows); // The mask for the bounding boxes
  for (int i = 0; i < nms_res.size(); ++i)
  {
    int idx = nms_res[i]; // index of the box to keep

    // Create the OutputSeg object and add it to the outputs vector
    OutputSeg result;
    result.class_id = class_ids[idx];
    result.confidence = confidences[idx];
    result.box = boxes[idx] & rect_bit_mask;
    result.mask = picked_proposals[idx];

    outputs.push_back(result);
  }

  for (int i = 0; i < outputs.size(); ++i)
  {
    // Generate the mask for each bounding box
    generateBoxMask(cv::Mat(outputs[i].mask).t(), net_output_img[1], outputs[i], conv_params, src_img.size());
  }

  // Generate the final mask
  cv::Mat gray_scale_mask = cv::Mat::zeros(src_img.size(), CV_8UC1); // The final mask
  int gray_level = 1;
  for (int i = 0; i < outputs.size(); i++)
  {
    // Recollecting where the bounding boxes were in the whole image
    int left = outputs[i].box.x;
    int top = outputs[i].box.y;

    // Adding the mask of the bounding box to the final mask
    gray_scale_mask(outputs[i].box).setTo(gray_level, outputs[i].box_mask);
    gray_level++; // Incrementing the gray level for the next object
  }

  gray_scale_mask.copyTo(final_mask); // Copying the final mask to the output
}

void Yolov8Seg::generateBoxMask(const cv::Mat &mask_candidate, const cv::Mat &mask_protos, OutputSeg &output, cv::Vec4d &conv_params, const cv::Size &src_img_shape)
{
  cv::Rect box = output.box; // The bounding box

  // Compute the range of the mask with respect to the segmentation output
  int rang_x = floor((box.x * conv_params[0] + conv_params[2]) / this->net_size.width * this->seg_width);
  int rang_y = floor((box.y * conv_params[1] + conv_params[3]) / this->net_size.height * this->seg_height);
  int rang_w = ceil(((box.x + box.width) * conv_params[0] + conv_params[2]) / this->net_size.width * this->seg_width) - rang_x;
  int rang_h = ceil(((box.y + box.height) * conv_params[1] + conv_params[3]) / this->net_size.height * this->seg_height) - rang_y;

  // Ensure that the range is within the segmentation output
  rang_w = MAX(rang_w, 1);
  rang_h = MAX(rang_h, 1);
  if (rang_x + rang_w > seg_width)
  {
    if (this->seg_width - rang_x > 0)
      rang_w = this->seg_width - rang_x;
    else
      rang_x -= 1;
  }
  if (rang_y + rang_h > this->seg_height)
  {
    if (this->seg_height - rang_y > 0)
      rang_h = this->seg_height - rang_y;
    else
      rang_y -= 1;
  }

  std::vector<cv::Range> roi_rangs; // The range of the mask in the segmentation output
  roi_rangs.push_back(cv::Range(0, 1)); // Only one
  roi_rangs.push_back(cv::Range::all()); // All channels
  roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y)); // The range of the mask
  roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x)); // The range of the mask

  // Cropping the mask from the segmentation output
  // based on the parameters computed above
  cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
  cv::Mat protos = temp_mask_protos.reshape(0, {this->seg_channels, rang_w * rang_h}); // Reshaping the mask to be of shape [seg_channels, rang_w * rang_h]
  cv::Mat matmul_res = (mask_candidate * protos).t(); // Multiplying the mask with the mask prototypes
  cv::Mat masks_feature = matmul_res.reshape(1, {rang_h, rang_w}); // Reshaping the result to be of shape [rang_h, rang_w]

  // Applying the sigmoid function to the mask
  cv::Mat dest, mask;
  cv::exp(-masks_feature, dest);
  dest = 1.0 / (1.0 + dest);

  // Computing position and size of the mask with respect to the image
  int left = floor((this->net_size.width / this->seg_width * rang_x - conv_params[2]) / conv_params[0]);
  int top = floor((this->net_size.height / this->seg_height * rang_y - conv_params[3]) / conv_params[1]);
  int width = ceil(this->net_size.width / this->seg_width * rang_w / conv_params[0]);
  int height = ceil(this->net_size.height / this->seg_height * rang_h / conv_params[1]);

  cv::resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST); // Resizing the mask to the size of the bounding box
  mask = mask(box - cv::Point(left, top)) > this->mask_threshold; // Thresholding the mask
  output.box_mask = mask; // Saving the mask
}

void Yolov8Seg::preProcessImage(const cv::Mat &src_img, cv::Mat &net_input_img, cv::Vec4d &conv_params)
{
  cv::Size img_size = src_img.size();

  // Calculate the ratio between the net size and the image size as the ratio between the width and the height
  // pick the smallest ratio to make sure that the image fits in the net size
  float ratio = std::min((float)this->net_size.height / (float)img_size.height,
                         (float)this->net_size.width / (float)img_size.width);

  int new_height = (int)std::round((float)img_size.height * ratio); // The new height of the image
  int new_width = (int)std::round((float)img_size.width * ratio);   // The new width of the image

  conv_params[0] = (float)new_width / (float)img_size.width;   // Saving the ratio between the net size and the image size
  conv_params[1] = (float)new_height / (float)img_size.height; // Saving the ratio between the net size and the image size

  float pad_w = (float)(this->net_size.width - new_width);   // Saving the padding width
  float pad_h = (float)(this->net_size.height - new_height); // Saving the padding height

  pad_w /= 2.0f; // Padding width for each side
  pad_h /= 2.0f; // Padding height for each side

  cv::Mat resized_img = cv::Mat::zeros(this->net_size, src_img.type()); // The resized image
  cv::resize(src_img, resized_img, cv::Size(new_width, new_height));    // Resize the image

  // Computing the padding for each side
  int top = int(std::round(pad_h - 0.1f));
  int bottom = int(std::round(pad_h + 0.1f));
  int left = int(std::round(pad_w - 0.1f));
  int right = int(std::round(pad_w + 0.1f));

  // Saving the padding for each side
  conv_params[2] = left;
  conv_params[3] = top;

  // Padding the image
  cv::copyMakeBorder(resized_img, resized_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  resized_img.copyTo(net_input_img); // Copying the image to the net input
}

cv::Mat Yolov8Seg::displaySegOutput(const cv::Mat &src_img, const cv::Mat &final_mask)
{
  // Checking if the src img is empty
  if (src_img.empty())
  {
    throw std::invalid_argument("The source image is empty");
  }
  // Checking the final mask is CV_8UC1
  if (final_mask.type() != CV_8UC1)
  {
    throw std::invalid_argument("The final mask is not CV_8UC1, have you run the runSegmentation function?");
  }
  std::map<uchar, cv::Vec3b> color_map; // The color map for the different objects
  std::vector<uchar> gray_levels; // The gray levels of the mask

  cv::Mat gray_mask = final_mask.clone();
  gray_mask.reshape(1, 1).copyTo(gray_levels); // Reshaping the mask to be of shape [1, width * height]

  for (int i = 0; i < gray_levels.size(); i++)
  {
    if (gray_levels[i] != 0)
    {
      // Generating a random color for each object
      color_map[gray_levels[i]] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
    }
  }

  cv::Mat colored_mask = cv::Mat::zeros(src_img.size(), CV_8UC3); // The colored mask
  for (int i = 0; i < src_img.rows; i++)
  {
    for (int j = 0; j < src_img.cols; j++)
    {
      if (gray_mask.at<uchar>(i, j) != 0)
      {
        // Coloring the mask
        colored_mask.at<cv::Vec3b>(i, j) = color_map[gray_mask.at<uchar>(i, j)];
      }
    }
  }

  return colored_mask;
}

void Yolov8Seg::runOnFolder(const std::string& folder_path, const std::string& output_folder_path) {
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
    cv::Mat src_img = cv::imread(file_names[i]);

    // Running the segmentation
    cv::Mat final_mask;
    runSegmentation(src_img, final_mask);

    // Displaying the output
    cv::Mat colored_mask = displaySegOutput(src_img, final_mask);

    cv::Mat img_with_mask = src_img.clone();
    img_with_mask = img_with_mask * 0.5 + colored_mask * 0.5;

    // Saving the output
    std::string output_file_name = output_folder_path + "/" + file_names[i].substr(file_names[i].find_last_of("/") + 1);
    cv::imwrite(output_file_name, img_with_mask);
  }
}