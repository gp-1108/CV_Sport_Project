/**
* @file performance.cpp
* @author Federico Gelain ID number: 2076737
* @date ---
* @version 1.0
*/

#include "performance.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <fstream>
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>

void playerLocalizationMetrics(std::string folderPredictionPath, std::string folderGroundTruthPath) {
    // Checking if the folders exist
    if (!cv::utils::fs::exists(folderPredictionPath)) {
        std::cout << "The folder" << folderPredictionPath.c_str() << "does not exist" << std::endl;
        return ;
    }
  
    if (!cv::utils::fs::exists(folderGroundTruthPath)) {
        std::cout << "The folder" << folderGroundTruthPath.c_str() << "does not exist" << std::endl;
        return ;
    }

    std::vector<cv::String> predictionFiles; // The names of the files in the folder
    cv::glob(folderPredictionPath, predictionFiles); // Getting the names of the files in the folder

    std::vector<cv::String> truthFiles; // The names of the files in the folder
    cv::glob(folderGroundTruthPath, truthFiles); // Getting the names of the files in the folder

  //std::printf("Detected %ld files in the folder %s\n", file_names1.size(), folder_path1.c_str());

  for (int i = 0; i < predictionFiles.size(); i++) {
    // Checking that the files are of the correct extension
    if (predictionFiles[i].find(".txt") == std::string::npos && truthFiles[i].find(".txt") == std::string::npos) {
      continue;
    }

    std::vector<std::pair<int, cv::Rect>> originalPredictedBoundingBoxes = getBoundingBoxesFromFile(predictionFiles[i], false);
    std::vector<std::pair<int, cv::Rect>> invertedPredictedBoundingBoxes = getBoundingBoxesFromFile(predictionFiles[i], true);
    std::vector<std::pair<int, cv::Rect>> trueBoundingBoxes = getBoundingBoxesFromFile(truthFiles[i], false);
    
    /*
    std::cout << "File1: " << file_names1[i] << ", File2: " << file_names2[i] << std::endl;
    for (int i = 0; i < bb1.size(); i++)
    {
      std::cout << bb1[i].first << "-" << bb1[i].second << ", " << bb2[i].first << "-" << bb2[i].second << std::endl;
    }*/

    double mAPNonInvertedLabels = mAPComputation(originalPredictedBoundingBoxes, trueBoundingBoxes);
    double mAPInvertedLabels = mAPComputation(invertedPredictedBoundingBoxes, trueBoundingBoxes);

    std::cout << "Non inverted: " << mAPNonInvertedLabels << ", inverted: " << mAPInvertedLabels << std::endl;

    std::cout << "Value taken:" << std::max(mAPNonInvertedLabels, mAPInvertedLabels) << std::endl;

    std::cout << std::endl;
  }
}


std::vector<std::pair<int, cv::Rect>> getBoundingBoxesFromFile(std::string filePath, bool inverted) {
  std::vector<std::pair<int, cv::Rect>> boundingBoxesInfo;
  
  std::ifstream bbFile (filePath);

  if(bbFile.is_open()) {
    std::string currLine;

    while(std::getline(bbFile, currLine)) {
      //handle the case in which the file is wrongly written and it contains empty lines
      if(currLine.size() > 0) {
        std::vector<int> bBoxData;

        std::stringstream sstr(currLine);

        std::string token;

        while(sstr >> token) {
          bBoxData.push_back(std::stoi(token));
        }

        /*
        for (int i = 0; i < bBoxData.size(); i++)
        {
          std::cout << bBoxData[i] << ", " << std::endl;
        }
        
        std::cout << std::endl;
        */

        if(inverted) {
            if(bBoxData[4] == 1)
                boundingBoxesInfo.push_back(std::make_pair(2, cv::Rect(bBoxData[0], bBoxData[1], bBoxData[2], bBoxData[3])));
            else
                boundingBoxesInfo.push_back(std::make_pair(1, cv::Rect(bBoxData[0], bBoxData[1], bBoxData[2], bBoxData[3])));
        }
        else 
            boundingBoxesInfo.push_back(std::make_pair(bBoxData[4], cv::Rect(bBoxData[0], bBoxData[1], bBoxData[2], bBoxData[3])));
      }
    }

    bbFile.close();
  }

  return boundingBoxesInfo;
}

double intersectionOverUnion(const cv::Rect prediction, const cv::Rect groundTruth) {
  //check for division by zero (if both areas are 0, the union will be 0)
  if(prediction.area() == 0 && groundTruth.area() == 0) {
    return 0.0;
  }

  /*std::cout << "Pred: " << prediction << std::endl;
  std::cout << "Truth: " << groundTruth << std::endl;*/

  /*
   * the idea is to create two binary masks big enough in which to draw the two rectangles. Then,
   * the intersection will be simply the multiplication of the two masks, while the union is the sum.
   * Using countNonZero it's possible to retrieve the number of pixels for both intersection and union
   * to compute the IoU value
   */
  int maxWidth = prediction.x + prediction.width + groundTruth.x + groundTruth.width;
  int maxHeight = prediction.y + prediction.height + groundTruth.y + groundTruth.height;

  // Create masks for the two rectangles
  cv::Mat maxAreaRect1(maxHeight, maxWidth, CV_8UC1, cv::Scalar(0));
  cv::Mat maxAreaRect2(maxHeight, maxWidth, CV_8UC1, cv::Scalar(0));

  cv::rectangle(maxAreaRect1, prediction, cv::Scalar(255), cv::FILLED);
  cv::rectangle(maxAreaRect2, groundTruth, cv::Scalar(255), cv::FILLED);

  //cv::imshow("Rect1", maxAreaRect1);
  //cv::imshow("Rect2", maxAreaRect2);

  cv::Mat unionMask = maxAreaRect1 + maxAreaRect2;
  cv::Mat intersectionMask = maxAreaRect1.mul(maxAreaRect2);

  //cv::imshow("Union", unionMask);
  //cv::imshow("Intersection", intersectionMask);

  //cv::waitKey(0);

  int numPixelsIntersection = cv::countNonZero(intersectionMask);
  int numPixelsUnion = cv::countNonZero(unionMask);

  //std::cout << "Iou: " << (double)numPixelsIntersection/numPixelsUnion << std::endl;
  return (double)numPixelsIntersection / numPixelsUnion;
}

double computeAP(std::vector<double> precision, std::vector<double> recall) {
    //AP is computed using 11 points interpolation. First thing to do, the precision-recall "plot" has to be sorted in increasing order based on the recall values computed before
    //(precision is the ordinate, recall the abscissa)
    std::vector<std::pair<double, double>> precisionRecallPlot;
    
    for(int j = 0; j < precision.size(); j++) {
      precisionRecallPlot.push_back(std::make_pair(precision[j], recall[j]));
    }

    std::sort(precisionRecallPlot.begin(), precisionRecallPlot.end(), [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
      return a.second < b.second;
    });

    /*
    std::cout << "\nPrecision-Recall (y, x) plot\n" << std::endl;
    for(int a = 0; a < precisionRecallPlot.size(); a++) {
      std::cout << "y: " << precisionRecallPlot[a].first << ", x: " << precisionRecallPlot[a].second << std::endl;
    }

    std::cout << std::endl;
    */

    std::vector<double> intervals;
    int interpolationIntervals = 11;

    //create the 11 recall intervals (from 0 to 1 with step 0.1)
    for(int k = 0; k < interpolationIntervals; k++) {
      intervals.push_back(0.1 * k);
    }

    std::vector<double> interpPrecision(interpolationIntervals, 0.0);

    /*
    std::cout << "Starting values of interpolated precision at each interval" << std::endl;
    for(int l = 0; l < interpPrecision.size(); l++) {
      std::cout << interpPrecision[l] << ", ";
      ap += interpPrecision[l];
    }*/

    /*
     * here is the computation of the interpolated precisions:
     * for each recall interval, compute the maximum precision considering only the pairs (precision, recall) that have
     * recall >= interval. To do so, it's easier to move from the end (since the pairs have been sorted in increasing order
     * based on the recalls) and go backwards until the true recall is smaller than the interval one
     */
    for(int a = 0; a < interpolationIntervals; a++) {
      int index = precisionRecallPlot.size() - 1;

      //while the true recall is bigger or equal then the interpolated one
      while(index >= 0 && precisionRecallPlot[index].second >= intervals[a]) {
        if(interpPrecision[a] < precisionRecallPlot[index].first) {
          interpPrecision[a] = precisionRecallPlot[index].first;
        }

        index--;
      }
    }

    /*
    std::cout << "\nInterpolated precision at each interval" << std::endl;
    for(int l = 0; l < interpPrecision.size(); l++) {
      std::cout << interpPrecision[l] << ", ";
      ap += interpPrecision[l];
    }*/

    return std::accumulate(interpPrecision.begin(), interpPrecision.end(), 0.0) / interpolationIntervals;
}

double mAPComputation(std::vector<std::pair<int, cv::Rect>> predictions, std::vector<std::pair<int, cv::Rect>> truth) {
  //since we know the bounding boxes only cover the players, the metrics are computed only for the labels team A and B (which are respectively 1 and 2, so we can index them by simply subtracting 1)
  //a structure is created to store, for each class, the recall and precision value at each iteration of all bounding boxes predicted
  std::vector<std::vector<double>> recall(2, std::vector<double>(0, 0));
  std::vector<std::vector<double>> precision(2, std::vector<double>(0, 0));

  //for each class, we keep track of the number of true positives and false positives to compute at each iteration both of the metrics 
  std::vector<int> cumulativeTruePositives(2, 0);
  std::vector<int> cumulativeFalsePositives(2, 0);
  std::vector<int> cumulativeFalseNegatives(2, 0);
  
  for(int i = 0; i < predictions.size(); i++) {
    int currentLabel = predictions[i].first;
    double maxIOUPositive = 0;
    double maxIOUNegative = 0;

    bool positive = false;
    //look through all the ground truth bounding boxes and compute the maximum Iou with the ones that have the same label of the predicted one
    for(int j = 0; j < truth.size(); j++) { 
      if(currentLabel == truth[j].first) {
        positive = true;
        double currIou = intersectionOverUnion(predictions[i].second, truth[j].second);

        if(maxIOUPositive < currIou) {
          maxIOUPositive = currIou;
        }
      }
      else {
        double currIou = intersectionOverUnion(predictions[i].second, truth[j].second);

        if(maxIOUNegative < currIou) {
          maxIOUNegative = currIou;
        }
      }
    }

    if(positive) { //at least one bounding box from the ground truth has the label predicted (so either TP or FP)
      //if the maximum iou computed is over the threshold, it means that the predicted bounding box has been classified correctly (true positive)
      if(maxIOUPositive > 0.5) {
        cumulativeTruePositives[currentLabel - 1]++;
      }
      else { //otherwise it's a false positive
        cumulativeFalsePositives[currentLabel - 1]++;
      }
    }
    else { //check if it is a false negative
      if(maxIOUNegative > 0.5) {
        cumulativeFalseNegatives[currentLabel - 1]++;
      }
    }

    //std::cout << "\nClass 1\nTP: " << cumulativeTruePositives[0] << ", FP: " << cumulativeFalsePositives[0] << std::endl;
    //std::cout << "\nClass 2\nTP: " << cumulativeTruePositives[1] << ", FP: " << cumulativeFalsePositives[1] << "\n" << std::endl;

    precision[currentLabel - 1].push_back((double)cumulativeTruePositives[currentLabel - 1] / (cumulativeTruePositives[currentLabel - 1] + cumulativeFalsePositives[currentLabel - 1]));
    recall[currentLabel - 1].push_back((double)cumulativeTruePositives[currentLabel - 1] / (cumulativeTruePositives[currentLabel - 1] + cumulativeFalseNegatives[currentLabel - 1]));
  }
 
  /*
  std::cout << "Class 1" << std::endl;
  for(int i = 0; i < precision.size(); i++) {
    std::cout << "Precision: " << precision[0][i] << ", recall: " << recall[0][i] << std::endl;
  }

  std::cout << "\nClass 2" << std::endl;
  for(int i = 0; i < precision.size(); i++) {
    std::cout << "Precision: " << precision[1][i] << ", recall: " << recall[1][i] << std::endl;
  }*/

  double mAP = 0.0;
  
  for(int i = 0; i < precision.size(); i++) {
    double ap = computeAP(precision[i], recall[i]);

    //std::cout << "\n\nAp for class " << i+1 << ": " << ap << std::endl;
    
    mAP += ap;
  }

  mAP /= precision.size();

  return mAP;
}