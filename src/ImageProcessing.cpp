#include "photo_booth/ImageProcessing.hpp"

#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

namespace photo_booth {

namespace {

void validateImage(const cv::Mat& image, const char* function_name) {
  if (image.empty()) {
    throw std::invalid_argument(std::string(function_name) +
                                ": input image is empty");
  }

  if (image.type() != CV_8UC3) {
    throw std::invalid_argument(std::string(function_name) +
                                ": input image must be CV_8UC3");
  }
}

}  // namespace

cv::Mat calcHist(const cv::Mat& image) {
  validateImage(image, "calcHist()");

  cv::Mat histogram = cv::Mat_<int>::zeros(3, 256);
  for (int row = 0; row < image.rows; ++row) {
    for (int column = 0; column < image.cols; ++column) {
      auto value = image.at<cv::Vec3b>(row, column);
      histogram.at<int>(0, value[0])++;
      histogram.at<int>(1, value[1])++;
      histogram.at<int>(2, value[2])++;
    }
  }

  return histogram;
}

cv::Mat swapRedBlueChannels(const cv::Mat& image) {
  validateImage(image, "swapRedBlueChannels()");

  cv::Mat output;

  cv::cvtColor(image, output, cv::COLOR_BGR2RGB);

  return output;
}

cv::Mat invertImage(const cv::Mat& image) {
  validateImage(image, "invertImage()");

  cv::Mat output;

  cv::bitwise_not(image, output);

  return output;
}


// My additions

//grayscale
cv::Mat enableGray(const cv::Mat& image) {
  validateImage(image, "enableGray()");  

  cv::Mat output;
  cv::Mat oneChannel;

  cv::cvtColor(image, oneChannel, cv::COLOR_BGR2GRAY);
  cv::cvtColor(oneChannel, output, cv::COLOR_GRAY2BGR);

  return output;
}

//uniform quantization
cv::Mat uniformQuant(const cv::Mat& image, const int& levels) {
  validateImage(image, "uniformQuant()");

  cv::Mat output = image.clone();
  for (int r=0; r<output.rows; r++) {
    for (int c=0; c<output.cols; c++) {
      cv::Vec3b& pixel = output.at<cv::Vec3b>(r,c);
      pixel[0] = (pixel[0] >> levels) << levels;
      pixel[1] = (pixel[1] >> levels) << levels;
      pixel[2] = (pixel[2] >> levels) << levels;
    }
  }

  return output;
}

//igs
cv::Mat improvedGS(const cv::Mat& image, const int& levels) {
  validateImage(image, "improvedGS()");

  cv::Mat output = image.clone();
  for (int r=0; r<output.rows; r++) {
    cv::Vec3b remainder(0, 0, 0);
    cv::Vec3b remainder_next(0, 0, 0);
    for (int c=0; c<output.cols; c++) {
      cv::Vec3b& pixel = output.at<cv::Vec3b>(r,c);

      //prevents wrap around
      for (int num=0; num<3; num++) {
	int pixel_wrap = pixel[num];
	int remainder_wrap = remainder[num];
	if (pixel_wrap + remainder_wrap < 255) {
	  pixel[num] = pixel[num] + remainder[num];
	}
	else {
	  pixel[num] = 255;
	}
      }

      remainder_next[0] = pixel[0] & ((1 << levels) - 1);
      remainder_next[1] = pixel[1] & ((1 << levels) - 1);
      remainder_next[2] = pixel[2] & ((1 << levels) - 1);

      pixel[0] = (pixel[0] >> levels) << levels;
      pixel[1] = (pixel[1] >> levels) << levels;
      pixel[2] = (pixel[2] >> levels) << levels;

      remainder = remainder_next;
    }
  }

  return output;
}

}  // namespace photo_booth
