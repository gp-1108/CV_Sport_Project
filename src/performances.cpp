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

    /*
    * In order to compare pairwise the predictions and the ground truth, the extraction of the files is done as follows:
    * 1) extract separately all files which ends with "bin.png" (the greyscale segmentation masks) and "bb.txt"
    *    (the bounding boxes);
    * 2) sort them in increasing order (the sort function extracts the image number and uses that to do the sorting);
    * 3) loop through each file (all the vectors have the same size so it's enough to loop through one of them and use
    *    the same index for all);
    * 4) compute the metrics
    */
    std::vector<cv::String> predictionFiles;
    cv::glob(folderPredictionPath, predictionFiles);

    std::vector<cv::String> predictionMasksFiles; //vector containing the path of the masks predicted
    std::vector<cv::String> predictionBBFiles; //vector containing the path of the bounding boxes predicted
    
    //fill the vectors with the correct kind of file paths
    for(int i = 0; i < predictionFiles.size(); i++) {
      if(predictionFiles[i].find("bin.png") != std::string::npos)
        predictionMasksFiles.push_back(predictionFiles[i]);

      if(predictionFiles[i].find("bb.txt") != std::string::npos)
        predictionBBFiles.push_back(predictionFiles[i]);
    }
    
    std::sort(predictionMasksFiles.begin(), predictionMasksFiles.end(), [](const std::string& a, const std::string& b) {
        //look for the last / in the file path
        size_t i = a.rfind('/', a.length());
        size_t j = b.rfind('/', b.length());
   
        //if there isn't, then the file is located in the same directory the program is run
        if (i == std::string::npos || j == std::string::npos) {
            size_t posA = fileNameA.find('_');
            size_t posB = fileNameB.find('_');
        
            //the files were named incorrectly. In this case simply compare the strings lexicographically
            if (posA == std::string::npos || posB == std::string::npos) {
              return fileNameA.compare(fileNameB) < 0;
            }

            //assumption that all files start with im, which is why the string is extracted starting from the third character
            std::string subA = fileNameA.substr(2, posA - 2);
            std::string subB = fileNameB.substr(2, posB - 2);
            
            //subA and subB both contain a number (assuming the files were named correctly). stoi solves the problem of file with 09,
            //converting it in 9 so that it can be easily compared
            int numA = std::stoi(subA);
            int numB = std::stoi(subB);

            //sort in increasing order of file number
            return numA < numB;
        }
        
        //this removes the path of the file, leaving only its name (and extension)
        std::string fileNameA = a.substr(i + 1, a.length() - i);
        std::string fileNameB = b.substr(j + 1, b.length() - j);

        size_t posA = fileNameA.find('_');
        size_t posB = fileNameB.find('_');
    
        //the files were named incorrectly. In this case simply compare the strings lexicographically
        if (posA == std::string::npos || posB == std::string::npos) {
          return fileNameA.compare(fileNameB) < 0;
        }

        //assumption that all files start with im, which is why the string is extracted starting from the third character
        std::string subA = fileNameA.substr(2, posA - 2);
        std::string subB = fileNameB.substr(2, posB - 2);
        
        //subA and subB both contain a number (assuming the files were named correctly). stoi solves the problem of file with 09,
        //converting it in 9 so that it can be easily compared
        int numA = std::stoi(subA);
        int numB = std::stoi(subB);

        //sort in increasing order of file number
        return numA < numB;
    });

    std::sort(predictionBBFiles.begin(), predictionBBFiles.end(), [](const std::string& a, const std::string& b) {
        //look for the last / in the file path
        size_t i = a.rfind('/', a.length());
        size_t j = b.rfind('/', b.length());
   
        //if there isn't, then the file is located in the same directory the program is run
        if (i == std::string::npos || j == std::string::npos) {
            size_t posA = fileNameA.find('_');
            size_t posB = fileNameB.find('_');
        
            //the files were named incorrectly. In this case simply compare the strings lexicographically
            if (posA == std::string::npos || posB == std::string::npos) {
              return fileNameA.compare(fileNameB) < 0;
            }

            //assumption that all files start with im, which is why the string is extracted starting from the third character
            std::string subA = fileNameA.substr(2, posA - 2);
            std::string subB = fileNameB.substr(2, posB - 2);
            
            //subA and subB both contain a number (assuming the files were named correctly). stoi solves the problem of file with 09,
            //converting it in 9 so that it can be easily compared
            int numA = std::stoi(subA);
            int numB = std::stoi(subB);

            //sort in increasing order of file number
            return numA < numB;
        }
        
        //this removes the path of the file, leaving only its name (and extension)
        std::string fileNameA = a.substr(i + 1, a.length() - i);
        std::string fileNameB = b.substr(j + 1, b.length() - j);

        size_t posA = fileNameA.find('_');
        size_t posB = fileNameB.find('_');
    
        //the files were named incorrectly. In this case simply compare the strings lexicographically
        if (posA == std::string::npos || posB == std::string::npos) {
          return fileNameA.compare(fileNameB) < 0;
        }

        //assumption that all files start with im, which is why the string is extracted starting from the third character
        std::string subA = fileNameA.substr(2, posA - 2);
        std::string subB = fileNameB.substr(2, posB - 2);
        
        //subA and subB both contain a number (assuming the files were named correctly). stoi solves the problem of file with 09,
        //converting it in 9 so that it can be easily compared
        int numA = std::stoi(subA);
        int numB = std::stoi(subB);

        //sort in increasing order of file number
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
        //look for the last / in the file path
        size_t i = a.rfind('/', a.length());
        size_t j = b.rfind('/', b.length());
   
        //if there isn't, then the file is located in the same directory the program is run
        if (i == std::string::npos || j == std::string::npos) {
            size_t posA = fileNameA.find('_');
            size_t posB = fileNameB.find('_');
        
            //the files were named incorrectly. In this case simply compare the strings lexicographically
            if (posA == std::string::npos || posB == std::string::npos) {
              return fileNameA.compare(fileNameB) < 0;
            }

            //assumption that all files start with im, which is why the string is extracted starting from the third character
            std::string subA = fileNameA.substr(2, posA - 2);
            std::string subB = fileNameB.substr(2, posB - 2);
            
            //subA and subB both contain a number (assuming the files were named correctly). stoi solves the problem of file with 09,
            //converting it in 9 so that it can be easily compared
            int numA = std::stoi(subA);
            int numB = std::stoi(subB);

            //sort in increasing order of file number
            return numA < numB;
        }
        
        //this removes the path of the file, leaving only its name (and extension)
        std::string fileNameA = a.substr(i + 1, a.length() - i);
        std::string fileNameB = b.substr(j + 1, b.length() - j);

        size_t posA = fileNameA.find('_');
        size_t posB = fileNameB.find('_');
    
        //the files were named incorrectly. In this case simply compare the strings lexicographically
        if (posA == std::string::npos || posB == std::string::npos) {
          return fileNameA.compare(fileNameB) < 0;
        }

        //assumption that all files start with im, which is why the string is extracted starting from the third character
        std::string subA = fileNameA.substr(2, posA - 2);
        std::string subB = fileNameB.substr(2, posB - 2);
        
        //subA and subB both contain a number (assuming the files were named correctly). stoi solves the problem of file with 09,
        //converting it in 9 so that it can be easily compared
        int numA = std::stoi(subA);
        int numB = std::stoi(subB);

        //sort in increasing order of file number
        return numA < numB;
    });

  std::sort(truthBBFiles.begin(), truthBBFiles.end(), [](const std::string& a, const std::string& b) {
        //look for the last / in the file path
        size_t i = a.rfind('/', a.length());
        size_t j = b.rfind('/', b.length());
   
        //if there isn't, then the file is located in the same directory the program is run
        if (i == std::string::npos || j == std::string::npos) {
            size_t posA = fileNameA.find('_');
            size_t posB = fileNameB.find('_');
        
            //the files were named incorrectly. In this case simply compare the strings lexicographically
            if (posA == std::string::npos || posB == std::string::npos) {
              return fileNameA.compare(fileNameB) < 0;
            }

            //assumption that all files start with im, which is why the string is extracted starting from the third character
            std::string subA = fileNameA.substr(2, posA - 2);
            std::string subB = fileNameB.substr(2, posB - 2);
            
            //subA and subB both contain a number (assuming the files were named correctly). stoi solves the problem of file with 09,
            //converting it in 9 so that it can be easily compared
            int numA = std::stoi(subA);
            int numB = std::stoi(subB);

            //sort in increasing order of file number
            return numA < numB;
        }
        
        //this removes the path of the file, leaving only its name (and extension)
        std::string fileNameA = a.substr(i + 1, a.length() - i);
        std::string fileNameB = b.substr(j + 1, b.length() - j);

        size_t posA = fileNameA.find('_');
        size_t posB = fileNameB.find('_');
    
        //the files were named incorrectly. In this case simply compare the strings lexicographically
        if (posA == std::string::npos || posB == std::string::npos) {
          return fileNameA.compare(fileNameB) < 0;
        }

        //assumption that all files start with im, which is why the string is extracted starting from the third character
        std::string subA = fileNameA.substr(2, posA - 2);
        std::string subB = fileNameB.substr(2, posB - 2);
        
        //subA and subB both contain a number (assuming the files were named correctly). stoi solves the problem of file with 09,
        //converting it in 9 so that it can be easily compared
        int numA = std::stoi(subA);
        int numB = std::stoi(subB);

        //sort in increasing order of file number
        return numA < numB;
    });
  
  //these vectors will allow to compute the overall average of the metrics
  std::vector<double> allMAP;
  std::vector<double> allMIoU;

  /*
   * For clarity purposes, computeMetrics writes a file in which, for each image, the value of both mAP and mIoU is shown.
   * It also shows the overall values of the metrics at the end.
  */
  std::ofstream metricsFile("metricsOutput.txt");

  metricsFile << "----- METRIC TABLE -----" << std::endl;
  metricsFile << "| IMAG | mAP   | mIoU  |" << std::endl;

  //loop through all the file paths
  for (int i = 0; i < predictionMasksFiles.size(); i++) {
    std::stringstream outputLine;    
    outputLine << "| ";

    std::string imageName = "im" + std::to_string(i+1);

    while(imageName.size() < 4)
      imageName += " ";

    outputLine << imageName << " | ";
    
    //************************** mAP computation ***********************************

    //for the prediction, since there's no assurance that the team labels assigned to the players are the exact same of the ones
    //found in the ground truth files, both cases are handled and at the end the highest value is kept
    std::vector<std::pair<int, cv::Rect>> originalPredictedBoundingBoxes = getBoundingBoxesFromFile(predictionBBFiles[i], false);
    std::vector<std::pair<int, cv::Rect>> invertedPredictedBoundingBoxes = getBoundingBoxesFromFile(predictionBBFiles[i], true);
    std::vector<std::pair<int, cv::Rect>> trueBoundingBoxes = getBoundingBoxesFromFile(truthBBFiles[i], false);

    //retreive the metric value for both cases
    double mAPNonInvertedLabels = mAPComputation(originalPredictedBoundingBoxes, trueBoundingBoxes);
    double mAPInvertedLabels = mAPComputation(invertedPredictedBoundingBoxes, trueBoundingBoxes);

    //keep the highest value
    double mAP = std::max(mAPNonInvertedLabels, mAPInvertedLabels);

    allMAP.push_back(mAP);

    //write the result in the output file
    std::stringstream roundedMetric;

    roundedMetric << std::setprecision(3) << mAP;

    std::string mAPStr = roundedMetric.str();

    //This is in case the number had less than 3 decimal places
    while(mAPStr.size() < 5)
      mAPStr += " ";

    outputLine << mAPStr << " | ";

    //************************** mIoU computation ***********************************

    //As before, for the predictions we compute the metrics considering the two cases for teams label
    cv::Mat originalPredictedSegmentationImage = cv::imread(predictionMasksFiles[i], cv::IMREAD_GRAYSCALE);
    cv::Mat invertedPredictedSegmentationImage = originalPredictedSegmentationImage.clone();
    cv::Mat trueSegmentationImage = cv::imread(truthMasksFiles[i], cv::IMREAD_GRAYSCALE);

    //swap the team labels
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

    //mIoU requires to compute IoU for each one of the 4 classes we have and then take the average
    std::vector<double> IoUPerClass;

    //compute both IoU (original and inverted) for each class and take the maximum
    for(int c = 0; c < 4; c++) {
      double IoUOriginal = intersectionOverUnionSegmentation(originalPredictedSegmentationImage, trueSegmentationImage, c);
      double IoUInverted = intersectionOverUnionSegmentation(invertedPredictedSegmentationImage, trueSegmentationImage, c);

      IoUPerClass.push_back(std::max(IoUOriginal, IoUInverted));
    }

    double mIoU = std::accumulate(IoUPerClass.begin(), IoUPerClass.end(), 0.0) / IoUPerClass.size();
    allMIoU.push_back(mIoU);

    //write the result in the output file
    roundedMetric.str(std::string());

    roundedMetric << std::setprecision(3) << mIoU;

    std::string mIoUStr = roundedMetric.str();

    //This is in case the number had less than 3 decimal places
    while(mIoUStr.size() < 5)
      mIoUStr += " ";

    outputLine << mIoUStr << " | ";

    metricsFile << outputLine.str() << std::endl;
  }

  //Compute the overall average metrics and write them in the file
  metricsFile << "------------------------\n" << std::endl;

  metricsFile << "--- OVERALL RESULTS ----" << std::endl;

  double meanMAP = std::accumulate(allMAP.begin(), allMAP.end(), 0.0) / allMAP.size();

  metricsFile << "Average mAP: " << std::setprecision(3) << meanMAP << std::endl;

  double meanMIoU = std::accumulate(allMIoU.begin(), allMIoU.end(), 0.0) / allMIoU.size();

  metricsFile << "Average mIoU: " << std::setprecision(3) << meanMIoU << std::endl;

  metricsFile.close();
}

double intersectionOverUnionSegmentation(const cv::Mat prediction, const cv::Mat groundTruth, int label) {
    /*
     * A breakdown of how this works:
     * 1) Create for prediction and ground truth a mask where a pixel which is assigned the class label will have value 255, 0 otherwise;
     * 2) Loop through the pixels of both masks together:
     *    if both masks have the pixel set to 255, then it counts for the intersection;
     *    if at least one mask has the pixel set to 255, then it counts for the union;
     * 3) At the end, compute IoU as the ratio of number of pixels counting for intersection and the ones counting for union
    */
    cv::Mat predictedMask(prediction.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat trueMask(prediction.size(), CV_8UC1, cv::Scalar(0));;

    //create the masks
    for(int i = 0; i < prediction.rows; i++) {
      for(int j = 0; j < prediction.cols; j++) {
        if(prediction.at<uchar>(i,j) == label)
          predictedMask.at<uchar>(i,j) = 255;
        else 
          predictedMask.at<uchar>(i,j) = 0;
      }
    }

    for(int i = 0; i < groundTruth.rows; i++) {
      for(int j = 0; j < groundTruth.cols; j++) {
        if(groundTruth.at<uchar>(i,j) == label)
          trueMask.at<uchar>(i,j) = 255;
        else
          trueMask.at<uchar>(i,j) = 0;
      }
    }

    int intersectionPixels = 0;
    int unionPixels = 0;

    //calculate the number of intersection and union pixels
    for(int i = 0; i < predictedMask.rows; i++) {
      for(int j = 0; j < predictedMask.cols; j++) {
        if(predictedMask.at<uchar>(i,j) == 255 && trueMask.at<uchar>(i,j) == 255)
          intersectionPixels++;

        if(predictedMask.at<uchar>(i,j) == 255 || trueMask.at<uchar>(i,j) == 255)
          unionPixels++;
      }
    }

    //this happens only if both masks are empty (extremely unlikely, but something to check regardless)
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
      //just in case that the file is wrongly written and it contains empty lines
      if(currLine.size() > 0) {
        /*
         * each line of the files should contain exactly 5 values, separated by " ":
         * the 4 parameters of the rectangle (x, y, width, height);
         * the label that indicates the team prediction
        */
        std::vector<int> bBoxData;

        std::stringstream sstr(currLine);

        std::string token;

        while(sstr >> token) {
          bBoxData.push_back(std::stoi(token));
        }

        //check that you retreived the correct number of parameters
        if(bBoxData.size() == 5) {
          //add the bounding box information in the structure
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
    //AP is computed using 11 points interpolation. First thing to do, create the precision-recall plot 
    std::vector<std::pair<double, double>> precisionRecallPlot;
    
    for(int j = 0; j < precision.size(); j++) {
      precisionRecallPlot.push_back(std::make_pair(precision[j], recall[j]));
    }

    /*
    std::cout << "\nPrecision-Recall (y, x) plot\n" << std::endl;
    for(int a = 0; a < precisionRecallPlot.size(); a++) {
      std::cout << "y: " << precisionRecallPlot[a].first << ", x: " << precisionRecallPlot[a].second << std::endl;
    }

    std::cout << std::endl;
    */

    //create the 11 recall intervals (from 0 to 1 with step 0.1)
    std::vector<double> intervals;
    int interpolationIntervals = 11;
    
    for(int k = 0; k < interpolationIntervals; k++) {
      intervals.push_back(0.1 * k);
    }

    std::vector<double> interpPrecision(interpolationIntervals, 0.0);

    /*
     * here is the computation of the interpolated precisions:
     * for each recall interval, compute the maximum precision considering only the pairs (precision, recall) that have
     * recall >= interval. To do so, it's easier to move from the end (since the pairs are sorted in increasing order
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

    //return AP as the average of all the interpolated precisions
    return std::accumulate(interpPrecision.begin(), interpPrecision.end(), 0.0) / interpolationIntervals;
}

double mAPComputation(std::vector<std::pair<int, cv::Rect>> predictions, std::vector<std::pair<int, cv::Rect>> truth) {
  //since we know the bounding boxes only cover the players, the metrics are computed only for the labels team A and B (which are respectively 1 and 2, so we can index them by simply subtracting 1)
  //a structure is created to store, for each class, the recall and precision value at each iteration of all bounding boxes predicted
  std::vector<int> numTruthPerClass(2, 0);
  
  //compute the number of ground truth boxes per class (needed for the recall computation)
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
 
  /*
   * having no information about the condifence each predicted bounding box has, the approach taken here is the following:
   * 1) loop through each ground truth bounding box;
   * 2) compute the IoU with each predicted bounding box that has the same label and isn't already assigned to another ground truth bounding box;
   * 3) if the IoU computed is above the threshold, than we have found a TP predicted bounding box. So check it so that no other ground truth bounding box can take it;
   * 4) otherwise, increase the number of FP (no predicted bounding box with the same label is close enough)
   * The computation of FN is not considered, since the recall is computed using the number of ground truth bounding boxes for each class instead
  */
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
          indexToRemove = j; //keep track of the best predicted bounding box found
        }
      }
    }

    if(maxIOUPositive > 0.5) { //was a TP found?
      cumulativeTruePositives[currentLabel - 1]++;
      alreadyTaken[indexToRemove] = true; //yes, so "remove" the predicted bounding box from further computations
    }
    else { //otherwise it's a false positive
      cumulativeFalsePositives[currentLabel - 1]++;
    }
    
    //update precision and recall accordingly. cumulativeTruePositives = 0 means that both precision and recall will be equal to 0. There's no need to store these values
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
  
  //for each class, compute the average precision
  for(int i = 0; i < precision.size(); i++) {
    double ap = computeAP(precision[i], recall[i]);

    //std::cout << "\n\nAp for class " << i+1 << ": " << ap << std::endl;
    
    mAP += ap;
  }

  //finally compute mAP as the average of all ap
  mAP /= precision.size();

  return mAP;
}