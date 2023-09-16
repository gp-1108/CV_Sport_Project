/**
* @file performances.cpp
* @author Federico Gelain ID number: 2076737
* @date ---
* @version 1.0
*/

#include "performances.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <fstream>
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

void computeMetrics(std::string folderPredictionPath, std::string folderGroundTruthPath) {
    // Checking if the folders exist
    if (!cv::utils::fs::exists(folderPredictionPath)) {
        std::cout << "The folder" << folderPredictionPath.c_str() << "does not exist" << std::endl;
        return ;
    }
  
    if (!cv::utils::fs::exists(folderGroundTruthPath)) {
        std::cout << "The folder" << folderGroundTruthPath.c_str() << "does not exist" << std::endl;
        return ;
    }

    std::vector<cv::String> predictionFiles;
    cv::glob(folderPredictionPath, predictionFiles);

    std::vector<cv::String> predictionMasksFiles;
    std::vector<cv::String> predictionBBFiles;
    
    for(int i = 0; i < predictionFiles.size(); i++) {
      if(predictionFiles[i].find("bin.png") != std::string::npos)
        predictionMasksFiles.push_back(predictionFiles[i]);

      if(predictionFiles[i].find("bb.txt") != std::string::npos)
        predictionBBFiles.push_back(predictionFiles[i]);
    }
    
    std::sort(predictionMasksFiles.begin(), predictionMasksFiles.end(), [](const std::string& a, const std::string& b) {
        size_t i = a.rfind('/', a.length());
        size_t j = b.rfind('/', b.length());
   
        if (i == std::string::npos || j == std::string::npos) {
            return a.compare(b) < 0;
        }
        
        std::string fileNameA = a.substr(i + 1, a.length() - i);
        std::string fileNameB = b.substr(j + 1, b.length() - j);

        size_t posA = fileNameA.find('_');
        size_t posB = fileNameB.find('_');
    
        std::string subA = fileNameA.substr(2, posA - 2);
        std::string subB = fileNameB.substr(2, posB - 2);
        
        int numA = std::stoi(subA);
        int numB = std::stoi(subB);

        return numA < numB;
    });

    std::sort(predictionBBFiles.begin(), predictionBBFiles.end(), [](const std::string& a, const std::string& b) {
        size_t i = a.rfind('/', a.length());
        size_t j = b.rfind('/', b.length());
   
        if (i == std::string::npos || j == std::string::npos) {
            return a.compare(b) < 0;
        }
        
        std::string fileNameA = a.substr(i + 1, a.length() - i);
        std::string fileNameB = b.substr(j + 1, b.length() - j);

        size_t posA = fileNameA.find('_');
        size_t posB = fileNameB.find('_');
    
        std::string subA = fileNameA.substr(2, posA - 2);
        std::string subB = fileNameB.substr(2, posB - 2);
        
        int numA = std::stoi(subA);
        int numB = std::stoi(subB);

        return numA < numB;
    });

    std::vector<cv::String> groundTruthFiles; // The names of the files in the folder
    cv::glob(folderGroundTruthPath, groundTruthFiles); // Getting the names of the files in the folder
    
    std::vector<cv::String> truthMasksFiles; // The names of the files in the folder
    std::vector<cv::String> truthBBFiles; // The names of the files in the folder
    
    for(int i = 0; i < groundTruthFiles.size(); i++) {
      if(groundTruthFiles[i].find("bin.png") != std::string::npos)
        truthMasksFiles.push_back(groundTruthFiles[i]);
      
      if(groundTruthFiles[i].find("bb.txt") != std::string::npos)
        truthBBFiles.push_back(groundTruthFiles[i]);
    }
    
    std::sort(truthMasksFiles.begin(), truthMasksFiles.end(), [](const std::string& a, const std::string& b) {
        size_t i = a.rfind('/', a.length());
        size_t j = b.rfind('/', b.length());
   
        if (i == std::string::npos || j == std::string::npos) {
            return a.compare(b) < 0;
        }
        
        std::string fileNameA = a.substr(i + 1, a.length() - i);
        std::string fileNameB = b.substr(j + 1, b.length() - j);

        size_t posA = fileNameA.find('_');
        size_t posB = fileNameB.find('_');
    
        std::string subA = fileNameA.substr(2, posA - 2);
        std::string subB = fileNameB.substr(2, posB - 2);
        
        int numA = std::stoi(subA);
        int numB = std::stoi(subB);

        return numA < numB;
    });

    std::sort(truthBBFiles.begin(), truthBBFiles.end(), [](const std::string& a, const std::string& b) {
        size_t i = a.rfind('/', a.length());
        size_t j = b.rfind('/', b.length());
   
        if (i == std::string::npos || j == std::string::npos) {
            return a.compare(b) < 0;
        }
        
        std::string fileNameA = a.substr(i + 1, a.length() - i);
        std::string fileNameB = b.substr(j + 1, b.length() - j);

        size_t posA = fileNameA.find('_');
        size_t posB = fileNameB.find('_');
    
        std::string subA = fileNameA.substr(2, posA - 2);
        std::string subB = fileNameB.substr(2, posB - 2);
        
        int numA = std::stoi(subA);
        int numB = std::stoi(subB);

        return numA < numB;
    });
  
  std::vector<double> allMAP;
  std::vector<double> allMIoU;

  std::ofstream metricsFile("metricsOutput.txt");

  metricsFile << "----- METRIC TABLE -----" << std::endl;
  metricsFile << "| IMAG | mAP   | mIoU  |" << std::endl;

  for (int i = 0; i < predictionMasksFiles.size(); i++) {
    std::stringstream outputLine;    
    outputLine << "| ";

    std::string imageName = "im" + std::to_string(i+1);

    while(imageName.size() < 4)
      imageName += " ";

    outputLine << imageName << " | ";
    
    //mAP computation
    std::vector<std::pair<int, cv::Rect>> originalPredictedBoundingBoxes = getBoundingBoxesFromFile(predictionBBFiles[i], false);
    std::vector<std::pair<int, cv::Rect>> invertedPredictedBoundingBoxes = getBoundingBoxesFromFile(predictionBBFiles[i], true);
    std::vector<std::pair<int, cv::Rect>> trueBoundingBoxes = getBoundingBoxesFromFile(truthBBFiles[i], false);

    double mAPNonInvertedLabels = mAPComputation(originalPredictedBoundingBoxes, trueBoundingBoxes);

    //std::cout << "Inverted" << std::endl;
    double mAPInvertedLabels = mAPComputation(invertedPredictedBoundingBoxes, trueBoundingBoxes);

    double mAP = std::max(mAPNonInvertedLabels, mAPInvertedLabels);

    allMAP.push_back(mAP);

    cv::Mat originalPredictedSegmentationImage = cv::imread(predictionMasksFiles[i], cv::IMREAD_GRAYSCALE);
    cv::Mat invertedPredictedSegmentationImage = originalPredictedSegmentationImage.clone();
    cv::Mat trueSegmentationImage = cv::imread(truthMasksFiles[i], cv::IMREAD_GRAYSCALE);

    //cv::Mat field(originalPredictedSegmentationImage.size(), CV_8UC1, cv::Scalar(0));

    for(int j = 0; j < originalPredictedSegmentationImage.rows; j++) {
      for(int k = 0; k < originalPredictedSegmentationImage.cols; k++) {
        if(originalPredictedSegmentationImage.at<uchar>(j,k) == 1)
          invertedPredictedSegmentationImage.at<uchar>(j,k) = 2;
        else {
          if(originalPredictedSegmentationImage.at<uchar>(j,k) == 2)
            invertedPredictedSegmentationImage.at<uchar>(j,k) = 1;
        }
      }
    }

    std::vector<double> IoUPerClass;

    for(int c = 0; c < 4; c++) {
      double IoUOriginal = intersectionOverUnionSegmentation(originalPredictedSegmentationImage, trueSegmentationImage, c);
      double IoUInverted = intersectionOverUnionSegmentation(invertedPredictedSegmentationImage, trueSegmentationImage, c);

      /*
      std::cout << "Class " << c << "\noriginal: " << IoUOriginal << ", inverted: " << IoUInverted << std::endl;
      std::cout << "maximum: " << std::max(IoUOriginal, IoUInverted) << std::endl;
      std::cout << std::endl;*/
      IoUPerClass.push_back(std::max(IoUOriginal, IoUInverted));
    }

    double mIoU = std::accumulate(IoUPerClass.begin(), IoUPerClass.end(), 0.0) / IoUPerClass.size();
    allMIoU.push_back(mIoU);

    std::stringstream roundedMetric;

    roundedMetric << std::setprecision(3) << mAP;

    std::string mAPStr = roundedMetric.str();

    //This is in case the number had less than 3 decimal places
    while(mAPStr.size() < 5)
      mAPStr += " ";

    outputLine << mAPStr << " | ";

    roundedMetric.str(std::string());

    roundedMetric << std::setprecision(3) << mIoU;

    std::string mIoUStr = roundedMetric.str();

    //This is in case the number had less than 3 decimal places
    while(mIoUStr.size() < 5)
      mIoUStr += " ";

    outputLine << mIoUStr << " | ";

    metricsFile << outputLine.str() << std::endl;
  }

  metricsFile << "------------------------\n" << std::endl;

  metricsFile << "--- OVERALL RESULTS ----" << std::endl;

  double meanMAP = std::accumulate(allMAP.begin(), allMAP.end(), 0.0) / allMAP.size();

  metricsFile << "Average mAP: " << std::setprecision(3) << meanMAP << std::endl;

  double meanMIoU = std::accumulate(allMIoU.begin(), allMIoU.end(), 0.0) / allMIoU.size();

  metricsFile << "Average mIoU: " << std::setprecision(3) << meanMIoU << std::endl;

  metricsFile.close();
}

double intersectionOverUnionSegmentation(const cv::Mat prediction, const cv::Mat groundTruth, int label) {
    cv::Mat predictedMask(prediction.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat trueMask(prediction.size(), CV_8UC1, cv::Scalar(0));;

    for(int i = 0; i < prediction.rows; i++) {
      for(int j = 0; j < prediction.cols; j++) {
        if(prediction.at<uchar>(i,j) == label)
          predictedMask.at<uchar>(i,j) = 255;
      }
    }

    for(int i = 0; i < groundTruth.rows; i++) {
      for(int j = 0; j < groundTruth.cols; j++) {
        if(groundTruth.at<uchar>(i,j) == label)
          trueMask.at<uchar>(i,j) = 255;
      }
    }

    int intersectionPixels = 0;
    int unionPixels = 0;

    for(int i = 0; i < predictedMask.rows; i++) {
      for(int j = 0; j < predictedMask.cols; j++) {
        if(predictedMask.at<uchar>(i,j) == 255 && trueMask.at<uchar>(i,j) == 255)
          intersectionPixels++;

        if(predictedMask.at<uchar>(i,j) == 255 || trueMask.at<uchar>(i,j) == 255)
          unionPixels++;
      }
    }

    if(unionPixels == 0)
      return 0.0;

    return (double)intersectionPixels/unionPixels;
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

double intersectionOverUnionBoundingBox(const cv::Rect prediction, const cv::Rect groundTruth) {
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

  cv::Mat intersectionMask = maxAreaRect1.mul(maxAreaRect2);
  cv::Mat unionMask = maxAreaRect1 + maxAreaRect2;

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
  std::vector<int> numTruthPerClass(2, 0);
  
  for(int i = 0; i < truth.size(); i++) {
    if(truth[i].first == 1)
      numTruthPerClass[0]++;
    else
      numTruthPerClass[1]++;
  }
  std::vector<std::vector<double>> recall(2, std::vector<double>(0, 0));
  std::vector<std::vector<double>> precision(2, std::vector<double>(0, 0));

  //for each class, we keep track of the number of true positives and false positives to compute at each iteration both of the metrics 
  std::vector<int> cumulativeTruePositives(2, 0);
  std::vector<int> cumulativeFalsePositives(2, 0);
  std::vector<int> cumulativeFalseNegatives(2, 0);
  
  std::vector<bool> alreadyTaken(predictions.size(), false);
 
  for(int i = 0; i < truth.size(); i++) {
    int currentLabel = truth[i].first;
    double maxIOUPositive = 0;
    double maxIOUNegative = 0;
    int indexToRemove = -1;

    //look through all the ground truth bounding boxes and compute the maximum Iou with the ones that have the same label of the predicted one
    for(int j = 0; j < predictions.size(); j++) { 
      if(!alreadyTaken[j] && currentLabel == predictions[j].first) {
        double currIou = intersectionOverUnionBoundingBox(predictions[j].second, truth[i].second);

        if(maxIOUPositive < currIou) {
          maxIOUPositive = currIou;
          indexToRemove = j;
        }
      }
    }

    if(maxIOUPositive > 0.5) {
      cumulativeTruePositives[currentLabel - 1]++;
      alreadyTaken[indexToRemove] = true;
    }
    else { //otherwise it's a false positive
      cumulativeFalsePositives[currentLabel - 1]++;
    }
    
    if(cumulativeTruePositives[currentLabel - 1] > 0) {
      precision[currentLabel - 1].push_back((double)cumulativeTruePositives[currentLabel - 1] / (cumulativeTruePositives[currentLabel - 1] + cumulativeFalsePositives[currentLabel - 1]));
      recall[currentLabel - 1].push_back((double)cumulativeTruePositives[currentLabel - 1] / (numTruthPerClass[currentLabel - 1]));
    }
  }

  /*
  std::cout << "Class 1" << std::endl;
  for(int i = 0; i < precision[0].size(); i++) {
    std::cout << "Precision: " << precision[0][i] << ", recall: " << recall[0][i] << std::endl;
  }

  std::cout << "\nClass 2" << std::endl;
  for(int i = 0; i < precision[1].size(); i++) {
    std::cout << "Precision: " << precision[1][i] << ", recall: " << recall[1][i] << std::endl;
  }
  */

  double mAP = 0.0;
  
  for(int i = 0; i < precision.size(); i++) {
    double ap = computeAP(precision[i], recall[i]);

    //std::cout << "\n\nAp for class " << i+1 << ": " << ap << std::endl;
    
    mAP += ap;
  }

  mAP /= precision.size();

  return mAP;
}