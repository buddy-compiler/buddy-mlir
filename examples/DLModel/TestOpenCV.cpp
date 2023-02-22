#include <iostream>
#include <opencv2/opencv.hpp>
int main() {
  cv::Mat inputImage = cv::imread("./YellowLabradorLooking_new.jpg");
  assert(inputImage.channels() == 3);
  printf("Build OpenCV success.\n");
}
