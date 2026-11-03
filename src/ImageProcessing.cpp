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
cv::Mat Grayscale(const cv::Mat& image) {
  validateImage(image, "Grayscale()");  

  cv::Mat output;
  cv::Mat oneChannel;

  cv::cvtColor(image, oneChannel, cv::COLOR_BGR2GRAY);
  cv::cvtColor(oneChannel, output, cv::COLOR_GRAY2BGR);

  return output;
}

//uniform quantization
cv::Mat uniform_Quantization(const cv::Mat& image, const int levels) {
  validateImage(image, "uniform_Quantization()");

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
cv::Mat improved_Grayscale(const cv::Mat& image, const int levels) {
  validateImage(image, "improved_Grayscale()");

  cv::Mat output = image.clone();
  for (int r=0; r<output.rows; r++) {
    cv::Vec3b remainder(0, 0, 0);
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

      remainder[0] = pixel[0] & ((1 << levels) - 1);
      remainder[1] = pixel[1] & ((1 << levels) - 1);
      remainder[2] = pixel[2] & ((1 << levels) - 1);

      pixel[0] = (pixel[0] >> levels) << levels;
      pixel[1] = (pixel[1] >> levels) << levels;
      pixel[2] = (pixel[2] >> levels) << levels;
    }
  }

  return output;
}

//color switch
cv::Mat colorSwitch(const cv::Mat& image, const int levels) {
  validateImage (image, "colorSwitch()");

  cv::Mat output = image.clone();
  for (int r=0; r<output.rows; r++) {
    for(int c=0; c<output.cols; c++) {
      cv::Vec3b& pixel = output.at<cv::Vec3b>(r,c);

      if (levels == 1) {
        pixel[1] = 0;
        pixel[2] = 0;
      }

      if (levels == 2) {
        pixel[0] = 0;
        pixel[2] = 0;
      }

      if (levels == 3) {
        pixel[0] = 0;
        pixel[1] = 0;
      }

      if (levels == 4) {
        pixel[2] = 0;
      }

      if (levels == 5) {
        pixel[1] = 0;
      }

      if (levels == 6) {
        pixel[0] = 0;
      }
    }
  }

  return output;
}

//histogram equalization
cv::Mat histogram_Equalization(const cv::Mat& image) {
  validateImage (image, "histogram_Equalization()");

  cv::Mat output = image.clone();
  int total_pixels = output.rows * output.cols;
  std::vector<cv::Mat> channels;
  cv::split(image, channels);

  for (int a=0; a<3; a++) {
    int hist[256] = {0};
    for (int r=0; r<channels[a].rows; r++) {
      for (int c=0; c<channels[a].cols; c++) {
        hist[channels[a].at<uchar>(r, c)]++;
      }
    }

    int cdf[256] = {0};
    cdf[0] = hist[0];

    for (int b=1; b<256; b++) {
      cdf[b] = cdf[b-1] + hist[b];
    }

    uchar lut[256];
    for (int l=0; l<256; l++) {
      int value = (cdf[l] * 255) / total_pixels;
      lut[l] = static_cast<uchar>(std::clamp(value, 0, 255));
    }

    cv::Mat& ch = channels[a];
    for (int r=0; r<ch.rows; r++) {
      for (int c=0; c<ch.cols; c++) {
        ch.at<uchar>(r, c) = lut[ch.at<uchar>(r, c)];
      }
    }
  }

  cv::merge(channels, output);
  return output;
}

//frequency domain filtering
void shiftDFT(cv::Mat& workingIMG) {
  workingIMG = workingIMG(cv::Rect(0, 0, workingIMG.cols & -2, workingIMG.rows & -2));
  int centerX = workingIMG.cols/2;
  int centerY = workingIMG.rows/2;

  cv::Mat q0(workingIMG, cv::Rect(0, 0, centerX, centerY));
  cv::Mat q1(workingIMG, cv::Rect(centerX, 0, centerX, centerY));
  cv::Mat q2(workingIMG, cv::Rect(0, centerY, centerX, centerY));
  cv::Mat q3(workingIMG, cv::Rect(centerX, centerY, centerX, centerY));

  cv::Mat temp;
  q0.copyTo(temp);
  q3.copyTo(q0);
  temp.copyTo(q3);
  q1.copyTo(temp);
  q2.copyTo(q1);
  temp.copyTo(q2);
}

cv::Mat generateFilterMask(const cv::Mat& image, const int type, const double cutoff, const int order) {
  int rows = image.rows;
  int cols = image.cols;

  cv::Mat filter = cv::Mat::zeros(rows, cols, CV_32FC2);

  double centerX = rows/2.0;
  double centerY = cols/2.0;

  //generate filter
  for(int a=0; a<rows; a++) {
    for (int b=0; b<cols; b++) {
      double dist = std::sqrt(std::pow(a-centerX, 2) + std::pow(b-centerY, 2));
      double value = 0.0;

      if (type == 1) {
	value = (dist <= cutoff) ? 1.0 : 0.0;
      }

      else if (type == 2) {
	value = (dist > cutoff) ? 1.0 : 0.0;
      }

      else if (type == 3) {
	value = 1.0/(1.0+std::pow(dist/cutoff, 2*order));
      }

      else if (type == 4) {
	value = std::exp(-std::pow(dist, 2)/(2.0*std::pow(cutoff, 2)));
      }

      filter.at<cv::Vec2f>(a, b) = cv::Vec2f(static_cast<float>(value), static_cast<float>(value));
    }
  }
  
  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  std::vector<cv::Mat> filteredChannels(3);

  //apply filter to all 3 channels
  for (int a=0; a<3; a++) {
    cv::Mat workIMG;
    channels[a].convertTo(workIMG, CV_32F);

    cv::Mat planes[] = {workIMG, cv::Mat::zeros(workIMG.size(), CV_32F)};
    cv::Mat complexIMG;
    cv::merge(planes, 2, complexIMG);

    cv::dft(complexIMG, complexIMG);

    shiftDFT(complexIMG);

    cv::mulSpectrums(complexIMG, filter, complexIMG, 0);

    shiftDFT(complexIMG);

    cv::Mat inverseIMG;
    cv::idft(complexIMG, inverseIMG, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE); 

    cv::normalize(inverseIMG, inverseIMG, 0, 255, cv::NORM_MINMAX, CV_8U);
    filteredChannels[a] = inverseIMG;
  }

  cv::Mat output;
  cv::merge(filteredChannels, output);
  return output;
}

//spatial filtering
cv::Mat spatialFilter(const cv::Mat& image, const cv::Mat& filterKernel) {
  cv::Mat output = image.clone();
  int rows = output.rows;
  int cols = output.cols;

  for (int x=0; x<rows; x++) {
    for (int y=0; y<cols; y++) {
      double sumBlue = 0.0;
      double sumGreen = 0.0;
      double sumRed = 0.0;

      for (int a=-1; a<=1; a++) {
	for (int b=-1; b<=1; b++) {
	  int neighborX = std::clamp(x+a, 0, rows-1);
	  int neighborY = std::clamp(y+b, 0, cols-1);

	  cv::Vec3b pixel = image.at<cv::Vec3b>(neighborX, neighborY);
	  double filter = filterKernel.at<double>(a+1, b+1);

	  sumBlue += pixel[0]*filter;
	  sumGreen += pixel[1]*filter;
	  sumRed += pixel[2]*filter;
        }
      }
      output.at<cv::Vec3b>(x, y) = cv::Vec3b(
	static_cast<uchar>(std::clamp(std::round(sumBlue), 0.0, 255.0)),
	static_cast<uchar>(std::clamp(std::round(sumGreen), 0.0, 255.0)),
	static_cast<uchar>(std::clamp(std::round(sumRed), 0.0, 255.0))
      );
    }
  }
  return output;
}

//affine transform
cv::Mat affineTransform(const cv::Mat& image, const double angle, const double scaleX, const double scaleY, const double translateX, const double translateY) {
  double radian = angle*(std::numbers::pi/180.0);
  double cos = std::cos(radian);
  double sin = std::sin(radian);

  cv::Mat matrix = (cv::Mat_<double>(2, 3) <<
    scaleX*cos, -scaleY*sin, translateX,
    scaleX*sin, scaleY*cos, translateY
    );

  cv::Mat output = image.clone();
  cv::warpAffine(image, output, matrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
  return output;
}

//log polar
cv::Mat logPolarTransform(const cv::Mat& image) {
  cv::Mat output = image.clone();

  cv::Mat mapX(output.size(), CV_32F);
  cv::Mat mapY(output.size(), CV_32F);

  int rows = output.rows;
  int cols = output.cols;

  double centerX = cols/2.0;
  double centerY = rows/2.0;

  double maxRad = std::sqrt(centerX*centerX + centerY*centerY);

  double scale = std::log(maxRad)/rows;

  for (int a=0; a<rows; a++) {
    double rho = a;
    double rad = std::exp(rho*scale);
    for (int b=0; b<cols; b++) {
      double theta = (double)b/(double)cols*2.0*std::numbers::pi;
      double x = centerX+rad*std::cos(theta);
      double y = centerY+rad*std::sin(theta);

      mapX.at<float>(a, b) = static_cast<float>(x);
      mapY.at<float>(a, b) = static_cast<float>(y);
    }
  }
  cv::remap(image, output, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  return output;
}

//perspective transform



}  // namespace photo_booth
