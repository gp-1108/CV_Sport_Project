#include "yolo.h"
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <vector>
#include <map>

Yolov8Seg::~Yolov8Seg(){}

Yolov8Seg::Yolov8Seg(std::string model_path)
{
  this->net = cv::dnn::readNet(model_path);
  this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
  this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void Yolov8Seg::runSegmentation(const cv::Mat &src_img, cv::Mat &final_mask)
{
  cv::Vec4d conv_params;
  cv::Mat net_input_img;
  preProcessImage(src_img, net_input_img, conv_params);

  cv::Mat blob;
  cv::dnn::blobFromImage(net_input_img, blob, 1 / 255.0, this->net_size, cv::Scalar(0, 0, 0), true, false);

  net.setInput(blob);
  std::vector<cv::Mat> net_output_img;
  std::vector<std::string> output_layer_names{"output0", "output1"};
  net.forward(net_output_img, output_layer_names); // get outputs

  std::vector<int> class_ids;                  // res-class_id
  std::vector<float> confidences;              // res-conf
  std::vector<cv::Rect> boxes;                 // res-box
  std::vector<std::vector<float>> picked_proposals; // output0[:,:, 4 + _className.size():net_width]===> for mask
  int net_width = this->classes.size() + 4 + 32;


  cv::Mat output0 = cv::Mat(cv::Size(net_output_img[0].size[2], net_output_img[0].size[1]), CV_32F, (float *)net_output_img[0].data).t(); //[bs,116,8400]=>[bs,8400,116]
  int rows = output0.rows;
  float *pdata = (float *)output0.data;


  for (int r = 0; r < rows; r++) {
    cv::Mat scores(1, this->classes.size(), CV_32FC1, pdata + 4);
    cv::Point classIdPoint;

    double max_class_score;
    cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
    max_class_score = (float) max_class_score;

    if (max_class_score >= this->class_threshold) {
      std::vector<float> temp_proto(pdata + 4 + this->classes.size(), pdata + net_width);
      picked_proposals.push_back(temp_proto);
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
	std::vector<int> nms_res;
	cv::dnn::NMSBoxes(boxes, confidences, this->class_threshold, this->nms_threshold, nms_res);

  std::vector<OutputSeg> outputs;

	cv::Rect rect_bit_mask(0, 0, src_img.cols, src_img.rows);
	for (int i = 0; i < nms_res.size(); ++i) {
		int idx = nms_res[i];

		OutputSeg result;
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & rect_bit_mask;
    result.mask = picked_proposals[idx];

    outputs.push_back(result);
	}

  for (int i = 0; i < outputs.size(); ++i) {
    generateBoxMask(cv::Mat(outputs[i].mask).t(), net_output_img[1], outputs[i], conv_params, src_img.size());
  }

  cv::Mat gray_scale_mask = cv::Mat::zeros(src_img.size(), CV_8UC1);
  int gray_level = 1;
  for (int i = 0; i < outputs.size(); i++) {
    int left = outputs[i].box.x;
    int top = outputs[i].box.y;
    gray_scale_mask(outputs[i].box).setTo(gray_level, outputs[i].box_mask);
    gray_level++;
  }

  gray_scale_mask.copyTo(final_mask);
}

void Yolov8Seg::generateBoxMask(const cv::Mat& mask_candidate, const cv::Mat& mask_protos, OutputSeg &output, cv::Vec4d& conv_params, const cv::Size& src_img_shape) {
  cv::Rect box = output.box;

	//crop from mask_protos
	int rang_x = floor((box.x * conv_params[0] + conv_params[2]) / this->net_size.width * this->seg_width);
	int rang_y = floor((box.y * conv_params[1] + conv_params[3]) / this->net_size.height * this->seg_height);
	int rang_w = ceil(((box.x + box.width) * conv_params[0] + conv_params[2]) / this->net_size.width * this->seg_width) - rang_x;
	int rang_h = ceil(((box.y + box.height) * conv_params[1] + conv_params[3]) / this->net_size.height * this->seg_height) - rang_y;

	rang_w = MAX(rang_w, 1);
	rang_h = MAX(rang_h, 1);
	if (rang_x + rang_w > seg_width) {
		if (this->seg_width - rang_x > 0)
			rang_w = this->seg_width - rang_x;
		else
			rang_x -= 1;
	}
	if (rang_y + rang_h > this->seg_height) {
		if (this->seg_height - rang_y > 0)
			rang_h = this->seg_height - rang_y;
		else
			rang_y -= 1;
	}

	std::vector<cv::Range> roi_rangs;
	roi_rangs.push_back(cv::Range(0, 1));
	roi_rangs.push_back(cv::Range::all());
	roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
	roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

	//crop
	cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
	cv::Mat protos = temp_mask_protos.reshape(0, { this->seg_channels, rang_w * rang_h });
	cv::Mat matmul_res = (mask_candidate * protos).t();
	cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
	cv::Mat dest, mask;

	//sigmoid
	cv::exp(-masks_feature, dest);
	dest = 1.0 / (1.0 + dest);

	int left = floor((this->net_size.width / this->seg_width * rang_x - conv_params[2]) / conv_params[0]);
	int top = floor((this->net_size.height / this->seg_height * rang_y - conv_params[3]) / conv_params[1]);
	int width = ceil(this->net_size.width / this->seg_width * rang_w / conv_params[0]);
	int height = ceil(this->net_size.height / this->seg_height * rang_h / conv_params[1]);

	cv::resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
	mask = mask(box - cv::Point(left, top)) > this->mask_threshold;
	output.box_mask = mask;
}

void Yolov8Seg::preProcessImage(const cv::Mat &src_img, cv::Mat &net_input_img, cv::Vec4d &conv_params)
{
  cv::Size img_size = src_img.size();

  float ratio = std::min((float)this->net_size.height / (float)img_size.height,
                         (float)this->net_size.width / (float)img_size.width);

  int new_height = (int)std::round((float)img_size.height * ratio);
  int new_width = (int)std::round((float)img_size.width * ratio);

  conv_params[0] = (float) new_width / (float) img_size.width;
  conv_params[1] = (float) new_height / (float) img_size.height;

  float pad_w = (float)(this->net_size.width - new_width);
  float pad_h = (float)(this->net_size.height - new_height);

  pad_w /= 2.0f;
  pad_h /= 2.0f;

  cv::Mat resized_img = cv::Mat::zeros(this->net_size, src_img.type());
  cv::resize(src_img, resized_img, cv::Size(new_width, new_height));

  int top = int(std::round(pad_h - 0.1f));
  int bottom = int(std::round(pad_h + 0.1f));
  int left = int(std::round(pad_w - 0.1f));
  int right = int(std::round(pad_w + 0.1f));

  conv_params[2] = left;
  conv_params[3] = top;

  cv::copyMakeBorder(resized_img, resized_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  resized_img.copyTo(net_input_img);
}

cv::Mat Yolov8Seg::displaySegOutput(const cv::Mat &src_img, const cv::Mat &final_mask) {
  std::map<uchar, cv::Vec3b> color_map;
  std::vector<uchar> gray_levels;
  
  cv::Mat gray_mask = final_mask.clone();
  gray_mask.reshape(1, 1).copyTo(gray_levels);

  for (int i = 0; i < gray_levels.size(); i++) {
    if (gray_levels[i] != 0) {
      color_map[gray_levels[i]] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
    }
  }

  cv::Mat colored_mask = cv::Mat::zeros(src_img.size(), CV_8UC3);
  for (int i = 0; i < src_img.rows; i++) {
    for (int j = 0; j < src_img.cols; j++) {
      if (gray_mask.at<uchar>(i, j) != 0) {
        colored_mask.at<cv::Vec3b>(i, j) = color_map[gray_mask.at<uchar>(i, j)];
      }
    }
  }

  return colored_mask;
}