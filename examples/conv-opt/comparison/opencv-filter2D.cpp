#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

void FillImage(cv::Mat img) {
  int i = 0;
  for (std::ptrdiff_t row = 0; row < img.rows; ++row)
    for (std::ptrdiff_t col = 0; col < img.cols; ++col)
    {
      img.at<uchar>(col, row) = i;
      if ((col + row) % 2)
        ++i;
    }
}

enum class boundaryOption {
  BORDER_CONSTANT,
  BORDER_REPLICATE,
  BORDER_REFLECT,
  BORDER_REFLECT101
};

void generateExpectedImage(cv::Mat img, cv::Mat expectedImg, int ddepth,
                           cv::Mat kernel, cv::Point anchor, double delta,
                           boundaryOption option, std::string name) {
  if (option == boundaryOption::BORDER_CONSTANT)
    cv::filter2D(img, expectedImg, ddepth, kernel, anchor, delta,
                 cv::BORDER_CONSTANT);
  // else if (option == boundaryOption::BORDER_REPLICATE)
  //   cv::filter2D(img, expectedImg, ddepth, kernel, anchor, delta,
  //                cv::BORDER_REPLICATE);
  // else if (option == boundaryOption::BORDER_REFLECT)
  //   cv::filter2D(img, expectedImg, ddepth, kernel, anchor, delta,
  //                cv::BORDER_REFLECT);
  // else if (option == boundaryOption::BORDER_REFLECT101)
  //   cv::filter2D(img, expectedImg, ddepth, kernel, anchor, delta,
  //                cv::BORDER_REFLECT101);

  cv::imwrite(name, expectedImg);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "Invalid CLI arguments\nUsage : ImageHeight ImageWidth "
                 "DestinationFolder\n";
  }

  cv::Mat origImg, expImg;
  origImg.create(atoi(argv[1]), atoi(argv[2]), CV_8UC1);
  expImg.create(atoi(argv[1]), atoi(argv[2]), CV_8UC1);

  FillImage(origImg);
  cv::imwrite("test_9x9.png", origImg);
  std::cout << origImg << "\n";

  double delta = 0;
  int ddepth = -1;
  cv::Point anchor;
  cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F);

  for (std::ptrdiff_t i = 0; i < kernel.rows; ++i) {
    for (std::ptrdiff_t j = 0; j < kernel.cols; ++j) {
      // Put all expected images inside a folder and then test them from there.
      std::string partialName;
      partialName = std::string(argv[3]) + "/ExpectedImage_" +
                    std::to_string(i) + "_" + std::to_string(j) + "_";
      anchor = cv::Point(i, j);

      generateExpectedImage(origImg, expImg, ddepth, kernel, anchor, delta,
                            boundaryOption::BORDER_CONSTANT,
                            partialName + "BORDER_CONSTANT.png");

      generateExpectedImage(origImg, expImg, ddepth, kernel, anchor, delta,
                            boundaryOption::BORDER_REPLICATE,
                            partialName + "BORDER_RELPICATE.png");

      generateExpectedImage(origImg, expImg, ddepth, kernel, anchor, delta,
                            boundaryOption::BORDER_REFLECT,
                            partialName + "BORDER_REFLECT.png");

      generateExpectedImage(origImg, expImg, ddepth, kernel, anchor, delta,
                            boundaryOption::BORDER_REFLECT101,
                            partialName + "BORDER_REFLECT101.png");
    }
  }
}
