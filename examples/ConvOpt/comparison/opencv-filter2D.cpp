#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

#include "../kernels.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty()) {
    cout << "could not load image.." << endl;
  }

  Mat kernel = Mat(3, 3, CV_32FC1, sobel3x3KernelAlign);
//   Mat kernel = Mat(5, 5, CV_32FC1, sobel5x5KernelAlign);
//   Mat kernel = Mat(7, 7, CV_32FC1, sobel7x7KernelAlign);
//   Mat kernel = Mat(9, 9, CV_32FC1, sobel9x9KernelAlign);
  Mat output;
  clock_t start,end;
  start = clock();
  filter2D(image, output, CV_32FC1, kernel);
  end = clock();
  cout << "Execution time: " 
       << (double)(end - start) / CLOCKS_PER_SEC << " s" << endl;
  
  // Choose a PNG compression level
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  bool result = false;
  try {
    result = imwrite(argv[2], output, compression_params);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;

  return 0;
}
