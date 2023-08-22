//===- DIPAll.cpp ---------------------------------------------------------===//
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
// All Buddy DIP functions.
//
//===----------------------------------------------------------------------===//

#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <opencv2/imgcodecs.hpp>

int main() {
  cv::Mat yutu =
      cv::imread("../../examples/images/YuTu.png", cv::IMREAD_GRAYSCALE);
  Img<float, 2> input(yutu);

  //===--------------------------------------------------------------------===//
  // Test function resize
  //===--------------------------------------------------------------------===//
  intptr_t outputSize[2] = {250, 100}; // {image_cols, image_rows}
  MemRef<float, 2> resizeOutput = dip::Resize2D(
      &input, dip::INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION,
      outputSize);
  cv::Mat imResizeOutput(resizeOutput.getSizes()[0], resizeOutput.getSizes()[1],
                         CV_32FC1, resizeOutput.getData());
  cv::imwrite("./dip_resize_nearest.png", imResizeOutput);

  //===--------------------------------------------------------------------===//
  // Test function rotate
  //===--------------------------------------------------------------------===//
  MemRef<float, 2> rotateOutputDegree =
      dip::Rotate2D(&input, 90, dip::ANGLE_TYPE::DEGREE);
  cv::Mat imRotateOutputDegree(rotateOutputDegree.getSizes()[0],
                               rotateOutputDegree.getSizes()[1], CV_32FC1,
                               rotateOutputDegree.getData());
  cv::imwrite("./dip_rotate_degree.png", imRotateOutputDegree);

  MemRef<float, 2> rotateOutputRadian =
      dip::Rotate2D(&input, M_PI / 2, dip::ANGLE_TYPE::RADIAN);
  cv::Mat imRotateOutputRadian(rotateOutputRadian.getSizes()[0],
                               rotateOutputRadian.getSizes()[1], CV_32FC1,
                               rotateOutputRadian.getData());
  cv::imwrite("./dip_rotate_radian.png", imRotateOutputRadian);

  //===--------------------------------------------------------------------===//
  // Test function correlation
  //===--------------------------------------------------------------------===//
  intptr_t sizesOutput[2] = {yutu.rows, yutu.cols};
  intptr_t sizesKernel[2] = {3, 3};
  MemRef<float, 2> corrReplicateOutput(sizesOutput);
  MemRef<float, 2> corrConstantOutput(sizesOutput);
  float kernelAlign[] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  MemRef<float, 2> kernel((float *)kernelAlign, sizesKernel);
  dip::Corr2D(&input, &kernel, &corrReplicateOutput, 0, 0,
              dip::BOUNDARY_OPTION::REPLICATE_PADDING);
  dip::Corr2D(&input, &kernel, &corrConstantOutput, 0, 0,
              dip::BOUNDARY_OPTION::CONSTANT_PADDING);
  cv::Mat imCorrOutputReplicate(corrReplicateOutput.getSizes()[0],
                                corrReplicateOutput.getSizes()[1], CV_32FC1,
                                corrReplicateOutput.getData());
  cv::Mat imCorrOutputConstant(corrConstantOutput.getSizes()[0],
                               corrConstantOutput.getSizes()[1], CV_32FC1,
                               corrConstantOutput.getData());
  cv::imwrite("./dip_corr_replicate.png", imCorrOutputReplicate);
  cv::imwrite("./dip_corr_constant.png", imCorrOutputConstant);
  return 0;
}
