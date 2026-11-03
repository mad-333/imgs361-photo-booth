#pragma once

#include <opencv2/core.hpp>

namespace photo_booth {

/**
 * @brief Calculates the histogram of each channel of an 8-bit BGR image.
 *
 * Channel histograms are returned in the rows of a 3x256 CV_32S.
 */
cv::Mat calcHist(const cv::Mat& image);

/**
 * @brief Swaps the blue and red channels of an 8-bit BGR image.
 */
cv::Mat swapRedBlueChannels(const cv::Mat& image);

/**
 * @brief Inverts every channel of an 8-bit BGR image.
 *
 * Each output channel value is 255 minus the corresponding input value.
 */
cv::Mat invertImage(const cv::Mat& image);


// My additions

//grayscale
cv::Mat Grayscale(const cv::Mat& image);

//uniform quantization
cv::Mat uniform_Quantization(const cv::Mat& image, const int levels);

//igs
cv::Mat improved_Grayscale(const cv::Mat& image, const int levels);

//color switch
cv::Mat colorSwitch(const cv::Mat& image, const int levels);

//histogram equalization
cv::Mat histogram_Equalization(const cv::Mat& image);

//frequency domain filtering
cv::Mat generateFilterMask(const cv::Mat& image, const int type, const double cutoff, const int order);

//spatial filtering
cv::Mat spatialFilter(const cv::Mat& image, const cv::Mat& filterKernel);

//affine transform
cv::Mat affineTransform(const cv::Mat& image, const double angle, const double scaleX, const double scaleY, const double translateX, const double translateY);

//log polar
cv::Mat logPolarTransform(const cv::Mat& image);

//perspective transform


}  // namespace photo_booth
