//====- perspectiveTransform.cpp - Example of buddy-opt tool =================//
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
// This file implements an example of perspective transformation using the
// perspective_transform operator. The visualization implementation has been
// modified from an OpenCV example. The perspective_transform operator will be
// compiled using the buddy-opt tool into an object file.
// This file will be linked with the object file to generate the executable
// file.
//
//===----------------------------------------------------------------------===//
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/imgcodecs/loadsave.h>
#include <opencv2/core.hpp>
#include <vector>
using namespace std;
using namespace cv;

static void onMouse(int event, int x, int y, int, void *);
Mat warping(Mat image, Size warped_image_size, vector<Point2f> srcPoints,
            vector<Point2f> dstPoints);

String windowTitle = "Perspective Transformation Demo";
String labels[4] = {"TL", "TR", "BR", "BL"};
vector<Point2f> roi_corners;
vector<Point2f> midpoints(4);
vector<Point2f> dst_corners(4);
int roiIndex = 0;
bool dragging;
int selected_corner_index = 0;
bool validation_needed = true;

int showImageInPopupWindow(int argc, char *argv[]) {

  Mat original_image = imread(argv[1], IMREAD_GRAYSCALE);
  Img<float, 2> input = dip::imread<float, 2>(argv[1], dip::IMGRD_GRAYSCALE);
  intptr_t inputHeigh = input.getSizes()[0];
  intptr_t inputWidth = input.getSizes()[1];
  Mat image;

  float original_image_cols = (float)original_image.cols;
  float original_image_rows = (float)original_image.rows;
  roi_corners.push_back(Point2f((float)(original_image_cols / 1.70),
                                (float)(original_image_rows / 4.20)));
  roi_corners.push_back(Point2f((float)(original_image.cols / 1.15),
                                (float)(original_image.rows / 3.32)));
  roi_corners.push_back(Point2f((float)(original_image.cols / 1.33),
                                (float)(original_image.rows / 1.10)));
  roi_corners.push_back(Point2f((float)(original_image.cols / 1.93),
                                (float)(original_image.rows / 1.36)));

  namedWindow(windowTitle, WINDOW_AUTOSIZE);
  namedWindow("Result", WINDOW_AUTOSIZE);
  moveWindow("Result", 1300, 20);
  moveWindow(windowTitle, 300, 20);

  setMouseCallback(windowTitle, onMouse, 0);

  bool endProgram = false;
  while (!endProgram) {
    if (validation_needed & (roi_corners.size() < 4)) {
      validation_needed = false;
      image = original_image.clone();

      for (size_t i = 0; i < roi_corners.size(); ++i) {
        circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);

        if (i > 0) {
          line(image, roi_corners[i - 1], roi_corners[(i)], Scalar(0, 0, 255),
               2);
          circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);
          putText(image, labels[i].c_str(), roi_corners[i],
                  FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
        }
      }
      imshow(windowTitle, image);
    }

    if (validation_needed & (roi_corners.size() == 4)) {
      image = original_image.clone();
      for (int i = 0; i < 4; ++i) {
        line(image, roi_corners[i], roi_corners[(i + 1) % 4], Scalar(0, 0, 255),
             2);
        circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);
        putText(image, labels[i].c_str(), roi_corners[i], FONT_HERSHEY_SIMPLEX,
                0.8, Scalar(255, 0, 0), 2);
      }

      imshow(windowTitle, image);

      midpoints[0] = (roi_corners[0] + roi_corners[1]) / 2;
      midpoints[1] = (roi_corners[1] + roi_corners[2]) / 2;
      midpoints[2] = (roi_corners[2] + roi_corners[3]) / 2;
      midpoints[3] = (roi_corners[3] + roi_corners[0]) / 2;

      dst_corners[0].x = 0;
      dst_corners[0].y = 0;
      dst_corners[1].x = (float)norm(midpoints[1] - midpoints[3]);
      dst_corners[1].y = 0;
      dst_corners[2].x = dst_corners[1].x;
      dst_corners[2].y = (float)norm(midpoints[0] - midpoints[2]);
      dst_corners[3].x = 0;
      dst_corners[3].y = dst_corners[2].y;

      vector<pair<intptr_t, intptr_t>> src;
      intptr_t x = dst_corners[1].x, y = dst_corners[2].y;
      vector<pair<intptr_t, intptr_t>> dst = {{0, 0}, {x, 0}, {x, y}, {0, y}};
      for (int i = 0; i < 4; i++) {
        src.push_back({(intptr_t)roi_corners[i].x, (intptr_t)roi_corners[i].y});
      }
      MemRef<float, 2> output = dip::PerspectiveTransform(&input, src, dst);

      cv::Mat warped_image(inputHeigh, inputWidth, CV_8UC1, Scalar(0));
      for (int h = 0; h < y; h++) {
        for (int w = 0; w < x; w++) {
          warped_image.at<uchar>(h, w) =
              static_cast<uchar>(output.getData()[h * inputWidth + w]);
        }
      }

      imshow("Result", warped_image);
    }

    char c = (char)waitKey(10);

    if ((c == 'q') | (c == 'Q') | (c == 27)) {
      endProgram = true;
    }

    if ((c == 'c') | (c == 'C')) {
      roi_corners.clear();
    }

    if ((c == 'r') | (c == 'R')) {
      roi_corners.push_back(roi_corners[0]);
      roi_corners.erase(roi_corners.begin());
    }

    if ((c == 'i') | (c == 'I')) {
      swap(roi_corners[0], roi_corners[1]);
      swap(roi_corners[2], roi_corners[3]);
    }
  }

  return 0;
}

static void onMouse(int event, int x, int y, int, void *) {
  // Action when left button is pressed
  if (roi_corners.size() == 4) {
    for (int i = 0; i < 4; ++i) {
      if ((event == EVENT_LBUTTONDOWN) && ((abs(roi_corners[i].x - x) < 10)) &&
          (abs(roi_corners[i].y - y) < 10)) {
        selected_corner_index = i;
        dragging = true;
      }
    }
  } else if (event == EVENT_LBUTTONDOWN) {
    roi_corners.push_back(Point2f((float)x, (float)y));
    validation_needed = true;
  }

  // Action when left button is released
  if (event == EVENT_LBUTTONUP) {
    dragging = false;
  }

  // Action when left button is pressed and mouse has moved over the window
  if ((event == EVENT_MOUSEMOVE) && dragging) {
    roi_corners[selected_corner_index].x = (float)x;
    roi_corners[selected_corner_index].y = (float)y;
    validation_needed = true;
  }
}

int main(int argc, char *argv[]) {
  showImageInPopupWindow(argc, argv);

  return 0;
}