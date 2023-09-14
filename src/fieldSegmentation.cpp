/**
* @file fieldSegmentation.h
* @author Federico Gelain ID number: 2076737
* @date ---
* @version 1.0
*/

#include "fieldSegmentation.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

cv::Mat fieldDetectionAndSegmentation(const cv::Mat fieldImage) {
    //first a mask is created so that the candidate colors are computed only in a region of interest of the image
    cv::Mat mask = cv::Mat(fieldImage.size(), CV_8UC1, 255);

    //the assumption made here is that the field will most certainly be present in the second half of the image (more or less),
    //meaning that everything above it can be considered as background
    for (int i = 0; i < 0.45 * fieldImage.rows; i++) {
        for (int j = 0; j < fieldImage.cols; j++) {
            mask.at<uchar>(i,j) = 0;
        }   
    }


    //2 works the best, in terms of accuracy and performance
    int peaksPerChannel = 2;
    std::vector<std::pair<cv::Vec3b, int>> candidateColors = computeCandidateColors(fieldImage, mask, peaksPerChannel);

    sortCandidateColors(fieldImage, mask, candidateColors);
    
    //Best parameters so far:
    //distanceThreshold = 60
    //areaThreshold = 0.15 * ...
    int distanceThreshold = 65; //threshold for the intensity Euclidean distance
    double areaThreshold = 0.15 * fieldImage.rows * fieldImage.cols; //arbitrary area requirement for a candidate color to be the field color

    cv::Mat fieldMask = computeFieldMask(fieldImage, candidateColors, distanceThreshold, areaThreshold);

    //WORK IN PROGRESS: for now it doesn't modify the image, but simply does some things that can be seen from the imshows
    //fieldPostProcessing(fieldMask);

    return fieldMask;
}

std::vector<std::pair<cv::Vec3b, int>> computeCandidateColors(cv::Mat fieldImage, cv::Mat mask, int l) {
    std::vector<cv::Mat> bgr_planes;
    cv::split(fieldImage, bgr_planes);
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;

    cv::calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, histRange, uniform, accumulate);

    //some blurring is applied in order to smooth the histograms, removing the least significative peaks
    cv::GaussianBlur(b_hist, b_hist, cv::Size(3, 3), 0, 0, cv::BORDER_CONSTANT);
    cv::GaussianBlur(g_hist, g_hist, cv::Size(3, 3), 0, 0, cv::BORDER_CONSTANT);
    cv::GaussianBlur(r_hist, r_hist, cv::Size(3, 3), 0, 0, cv::BORDER_CONSTANT);

    //code snippet that shows the three channel histograms. Uncomment for debugging if needed
    /*
        int hist_w = 512, hist_h = 512;
        int bin_w = cvRound((double)hist_w / histSize);
        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

        cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

        for (int i = 1; i < histSize; i++)
        {
            cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                cv::Scalar(255, 0, 0), 2, 8, 0);

            cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                cv::Scalar(0, 255, 0), 2, 8, 0);

            cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                cv::Scalar(0, 0, 255), 2, 8, 0);
        }

        cv::imshow("Histograms", histImage);
        cv::waitKey(0);
    */

    /*
     * now the goal is to compute the peaks in each one of the histograms. To do so, the following steps are done:
     * 1) compute each possible local peak, which by definition is the bin whose frequency value in the image is higher than the one of its predecessor and subsequent bin;
     * 2) sort the peaks computed in point 1) by their frequency in the image in decreasing order. Meaning that the dominant bin color will be in position 0, the second one in position 1 etc;
     * The smoothing operation done before gives some ensurance that the peaks found won't be extremely similar to each other, since originally there are plenty basically on each slope of the histogram.
     * Another assumption done here is that the color of the field won't be black or a color near it, meaning that the lowest bins are excluded from the peaks computation. A reasonable starting bin value is
     * around 20/25 for each channel, since higher values would exclude otherwise correct colors like teal, green or beige
    */

    const int startingBlueBin = 20;
    const int startingRedBin = 25;
    const int startingGreenBin = 20;

    std::vector<std::pair<int, float>> b_peaks;
    std::vector<std::pair<int, float>> r_peaks;
    std::vector<std::pair<int, float>> g_peaks;

    for (int i = startingRedBin; i < b_hist.rows - 1; i++) {
        if (b_hist.at<float>(i) > b_hist.at<float>(i - 1) && b_hist.at<float>(i) > b_hist.at<float>(i + 1)) {
            b_peaks.push_back(std::make_pair(i, b_hist.at<float>(i)));
        }
    }

    std::sort(b_peaks.begin(), b_peaks.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    });

    for (int i = startingRedBin; i < g_hist.rows - 1; i++) {
        if (g_hist.at<float>(i) > g_hist.at<float>(i - 1) && g_hist.at<float>(i) > g_hist.at<float>(i + 1)) {
            g_peaks.push_back(std::make_pair(i, g_hist.at<float>(i)));
        }
    }

    std::sort(g_peaks.begin(), g_peaks.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    });

    for (int i = startingGreenBin; i < r_hist.rows - 1; i++) {
        if (r_hist.at<float>(i) > r_hist.at<float>(i - 1) && r_hist.at<float>(i) > r_hist.at<float>(i + 1)) {
            r_peaks.push_back(std::make_pair(i, r_hist.at<float>(i)));
        }
    }

    std::sort(r_peaks.begin(), r_peaks.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    });

    //code snippet that shows for each channel where the peaks were detected, sorted by decreasing order. Uncomment for debugging if needed
    /*
        std::cout << "Detected blue peaks at bin indices:" << std::endl;
        for (const auto& peak : b_peaks) {
            std::cout << "Index: " << peak.first << ", value: " << peak.second << std::endl;
        }

        std::cout << "Detected grren peaks at bin indices:" << std::endl;
        for (const auto& peak : g_peaks) {
            std::cout << "Index: " << peak.first << ", value: " << peak.second << std::endl;
        }
        
        std::cout << "Detected red peaks at bin indices:" << std::endl;
        for (const auto& peak : r_peaks) {
            std::cout << "Index: " << peak.first << ", value: " << peak.second << std::endl;
        }
    */

    /*
     * now it's finally possible to compute the candidate colors. For each channel, the first l peaks are considered and each possible combination of the peaks
     * of each channel will generate a different color. For this implementation, l = 2 (meaning 8 possible candidates) provides a good solution maintaining good performance
    */

    //to each color is associated an int value, which represents the number of pixels in the image for which that color is the closest (in terms of distance)
    std::vector<std::pair<cv::Vec3b, int>> candidateColors;

    for(int i = 0; i < l; i++) {
        for(int j = 0; j < l; j++) {
            for(int k = 0; k < l; k++) {
    		    candidateColors.push_back(std::make_pair(cv::Vec3b(b_peaks[i].first, g_peaks[j].first, r_peaks[k].first), 0));
            }
        }
    }

    return candidateColors;
}

void sortCandidateColors(const cv::Mat fieldImage, cv::Mat mask, std::vector<std::pair<cv::Vec3b, int>>& candidateColors) {
    for (int i = 0; i < fieldImage.rows; i++) {
        for (int j = 0; j < fieldImage.cols; j++) {
            //if the pixel is outside the region of interest (the lower half ot the image), then skip it
            if(mask.at<uchar>(i,j) == 0)
                continue;
            
            //for each candidate color, compute the intensity Euclidean distance with respect to the pixel, and choose the color whose distance is the minimum one
            double min_distance = std::numeric_limits<double>::max();
            int closestCandidateIndex = -1;

            for (int k = 0; k < candidateColors.size(); k++) {
                double dist = 0.0;

                for (int ch = 0; ch < fieldImage.channels(); ch++) {
                    dist += std::pow((fieldImage.at<cv::Vec3b>(i, j)[ch] - candidateColors[k].first[ch]), 2);
                }

                dist = std::sqrt(dist);

                if (dist < min_distance) {
                    min_distance = dist;
                    closestCandidateIndex = k;
                }                
            }

            //increase by one the counter for the corresponding color
            candidateColors[closestCandidateIndex].second++;
        }
    }
    
    //like before, the colors are sorted in decreasing order based on the frequency value
    std::sort(candidateColors.begin(), candidateColors.end(), [](const std::pair<cv::Vec3b, int>& a, const std::pair<cv::Vec3b, int>& b) {
        return a.second > b.second;
    });
}

cv::Mat computeFieldMask(const cv::Mat fieldImage, std::vector<std::pair<cv::Vec3b, int>> candidateColors, int distanceThreshold, double areaThreshold) {
    /*
     * now comes the actual computation of the field color. For each one of the candidates computed before, a binary mask is computed in this way:
     * if the pixel in the image has an intensity Euclidean distance smaller than a certain threshold, then set its mask value to 255,
     * otherwise leave the default value 0.
     * After that, count the number of non zero pixels and see if they are at least an arbitrary fraction of the total image area (for this
     * implementation 15% of the total area, i.e. the 15% of the total number of pixels, is considered).
     * If the outcome is true, then the candidate field color has been found. If it's false, move to the next color until you either find one or you 
     * reach the end of the array. In that case, consider the one that yields the largest area.
    */
    cv::Mat binaryFieldMask(fieldImage.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat coloredFieldMask(fieldImage.size(), CV_8UC3, cv::Scalar(0,0,0));

    cv::Vec3b dominantColor; //value of the field color selected
    
    bool colorFound = false; //flag to stop once the field color (the one that satisfies the minimum area requirement) is found
    int index = 0; //index of the current candidate color
    int maxIndex = 0; //index of the color to assign if no candidate color satisfies the minimum area requirement
    int maxArea = 0; //area associated with the color of maxIndex

    /*
     cv::Point2d dominantPointCoordinates;
     int numDom = 0;
     int coordx = 0;
     int coordy = 0;
    */
    
    while(!colorFound && index < candidateColors.size()) {
        dominantColor = cv::Vec3b(candidateColors[index].first[0], candidateColors[index].first[1], candidateColors[index].first[2]);
	
	    //std::cout << (int) candidateColors[index].first[0] << ", " << (int)candidateColors[index].first[1] << ", " << (int) candidateColors[index].first[2] << std::endl;
        
        for (int i = 0; i < coloredFieldMask.rows; i++) {
            for (int j = 0; j < coloredFieldMask.cols; j++) {    
                double dist = 0;

                for (int ch = 0; ch < coloredFieldMask.channels(); ch++) {
                    dist += std::pow((candidateColors[index].first[ch] - fieldImage.at<cv::Vec3b>(i,j)[ch]), 2);
                }

                dist = std::sqrt(dist);
                
                if(dist < distanceThreshold){
                    coloredFieldMask.at<cv::Vec3b>(i,j) = dominantColor;
                    binaryFieldMask.at<uchar>(i,j) = 3;
                    
                    //coordx += i;
                    //coordy += j;
                    //numDom++;
                }
            }
        }
        
        int nonZeroPixels = cv::countNonZero(binaryFieldMask);
        
        //if the number of non zero pixels is below the threshold
        if(nonZeroPixels < areaThreshold) {
            //then reset both images to their default values
            coloredFieldMask.setTo(cv::Scalar(0,0,0));
            binaryFieldMask.setTo(cv::Scalar(0));
            
            index++; //move to the next color
            
            //keep track of the color that yields the biggest area (in case no color satisfies the threshold requirement)
            if(maxArea < nonZeroPixels) {
                maxArea = nonZeroPixels;
                maxIndex = index;
            }

            //numDom = 0;
            //coordx = 0;
            //coordy = 0;
        }
        else {
            colorFound = true; //the color has been found and so the while loop can stop
        }
    }

    /*
     * if that's true, it means that no candidate color satisfies the thresholArea requirement. In this case, consider the
     * color associated with maxIndex and create the corresponding mask associated with it
    */
    if(index == candidateColors.size()) {
        dominantColor = candidateColors[maxIndex].first;
        
        for (int i = 0; i < coloredFieldMask.rows; i++) {
            for (int j = 0; j < coloredFieldMask.cols; j++) {
                double dist = 0;

                for (int ch = 0; ch < coloredFieldMask.channels(); ch++) {
                    dist += std::pow((candidateColors[maxIndex].first[ch] - coloredFieldMask.at<cv::Vec3b>(i,j)[ch]), 2);
                }

                dist = std::sqrt(dist);
            
                if(dist < distanceThreshold){
                    coloredFieldMask.at<cv::Vec3b>(i,j) = dominantColor;
                    binaryFieldMask.at<uchar>(i,j) = 3;
                    
                    //coordx += i;
                    //coordy += j;
                    //numDom++;
                }
            }
        }
    }

    //cv::imshow("Colored field mask", coloredFieldMask);
    //cv::waitKey(0);

    return binaryFieldMask;
}

void postProcessing(cv::Mat& fieldImage) {
    //first create the binary mask of the image, so that it's possible to apply morphological operations and such directly
    cv::Mat binaryFieldImage = (fieldImage != cv::Vec3b(0, 0, 0));
    binaryFieldImage = binaryFieldImage * 255;

    cv::cvtColor(binaryFieldImage, binaryFieldImage, cv::COLOR_BGR2GRAY);

    cv::imshow("Before anything", binaryFieldImage);
    cv::waitKey(0);

    //apply opening, to remove all small details which aren't needed
    double alpha = 1.0;
    int diameter = static_cast<int>((alpha / 100) * std::sqrt(std::pow(binaryFieldImage.rows, 2) + std::pow(binaryFieldImage.cols, 2)));
    cv::Mat structElem = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(diameter, diameter));

    cv::Mat openedImage;
    cv::morphologyEx(binaryFieldImage, openedImage, cv::MORPH_OPEN, structElem);

    cv::imshow("Image after opening", openedImage);
    cv::waitKey(0);

    //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
    std::vector<std::vector<cv::Point>> contours;
    
    cv::findContours(openedImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    //Draw the contours
    cv::Mat contourImage(openedImage.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Scalar colors[3];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
    for (int i = 0; i < contours.size(); i++) {
        cv::drawContours(contourImage, contours, i, colors[i % 3], cv::FILLED);
        cv::fillPoly(contourImage, contours[i], colors[(i + 1) % 3]);
        
        if(cv::contourArea(contours[i]) < 0.5/100*(binaryFieldImage.rows * binaryFieldImage.cols)) {
            uchar nearColor;
            cv::Mat mask(binaryFieldImage.size(), CV_8UC1, cv::Scalar(0));
            
            cv::drawContours(mask, contours, i, 255, cv::FILLED);
        
            //cv::imshow("Mask", mask);
            //cv::waitKey(0);
            
            double mean = cv::mean(binaryFieldImage, mask)[0];
            
            if(mean < 127) //so the mean value was 0
                cv::drawContours(openedImage, contours, i, 255, cv::FILLED);
            else
                cv::drawContours(openedImage, contours, i, 0, cv::FILLED);
        }
    }

    cv::imshow("Contours", contourImage);
    cv::waitKey(0);

    cv::imshow("Result", openedImage);
    cv::waitKey(0);
}