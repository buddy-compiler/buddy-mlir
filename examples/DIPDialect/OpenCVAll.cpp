//===- OpenCVAll.cpp ------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// OpenCV functions corresponding to Buddy DIP functions.
//
//===----------------------------------------------------------------------===//

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
  cv::Mat ocvInputImage =
      cv::imread("../../examples/images/YuTu.png", cv::IMREAD_GRAYSCALE);

  //===--------------------------------------------------------------------===//
  // Test function resize
  //===--------------------------------------------------------------------===//
  cv::Mat ocvOutputResize;
  cv::resize(ocvInputImage, ocvOutputResize, cv::Size(250, 100), 0, 0,
             cv::INTER_NEAREST);
  imwrite("opencv_resize2d.png", ocvOutputResize);

  //===--------------------------------------------------------------------===//
  // Test function rotate
  //===--------------------------------------------------------------------===//
  cv::Mat ocvOutputRotateDegree;
  // Define the rotation angle in degrees
  double angle = 270.0;
  // Get the image dimensions
  int height = ocvInputImage.rows;
  int width = ocvInputImage.cols;
  cv::Point2f center(width / 2.0, height / 2.0);
  cv::Mat rotationMatrixDegree = cv::getRotationMatrix2D(center, angle, 1.0);
  cv::warpAffine(ocvInputImage, ocvOutputRotateDegree, rotationMatrixDegree,
                 cv::Size(width, height));
  imwrite("opencv_rotate_degree.png", ocvOutputRotateDegree);

  cv::Mat ocvOutputRotateRadian;
  double angleRad = (3.0 / 2.0) * M_PI; // 270 degrees in radians
  cv::Mat rotationMatrixRadian =
      cv::getRotationMatrix2D(center, angleRad * 180.0 / M_PI, 1.0);
  cv::warpAffine(ocvInputImage, ocvOutputRotateRadian, rotationMatrixRadian,
                 cv::Size(width, height));
  imwrite("opencv_rotate_radian.png", ocvOutputRotateRadian);

  //===--------------------------------------------------------------------===//
  // Test function filter
  //===--------------------------------------------------------------------===//
  float kernelAlign[] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  cv::Mat ocvKernelFilter2D = Mat(3, 3, CV_32FC1, kernelAlign);
  Mat ocvOutputFilter2DConstant;
  filter2D(ocvInputImage, ocvOutputFilter2DConstant, CV_32FC1,
           ocvKernelFilter2D, cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);

  imwrite("opencv_filter2d_constant.png", ocvOutputFilter2DConstant);

  Mat ocvOutputFilter2DReplicate;
  filter2D(ocvInputImage, ocvOutputFilter2DReplicate, CV_32FC1,
           ocvKernelFilter2D, cv::Point(0, 0), 0.0, cv::BORDER_REPLICATE);
  imwrite("opencv_filter2d_replicate.png", ocvOutputFilter2DReplicate);

  return 0;
}
