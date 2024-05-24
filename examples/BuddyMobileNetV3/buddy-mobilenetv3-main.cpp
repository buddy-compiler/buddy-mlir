//===- MobileNetBenchmark.cpp ---------------------------------------------===//
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

#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

constexpr size_t ParamsSize = 2554968;
const std::string ImgName = "dog.png";

// Declare the mobilenet C interface.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                          MemRef<float, 1> *arg0,
                          MemRef<long long, 1> *arg1,
                          Img<float, 4> *input);

const cv::Mat imagePreprocessing() {
  // Get the directory of the LeNet example and construct the image path.
  std::string mobilenetDir = getenv("MOBILENETV3_EXAMPLE_PATH");
  std::string imgPath = mobilenetDir + "/images/" + ImgName; 
  // Read the image in grayscale mode.
  cv::Mat inputImage = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
  assert(!inputImage.empty() && "Could not read the image.");
  cv::Mat resizedImage;
  int imageWidth = 224;
  int imageHeight = 224;
  // Resize the image to 224x224 pixels.
  cv::resize(inputImage, resizedImage, cv::Size(imageWidth, imageHeight),
             cv::INTER_LINEAR);
  return resizedImage;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

void loadParameters(const std::string &floatParamPath,
                    const std::string &int64ParamPath,
                    MemRef<float, 1> &floatParam,
                    MemRef<long long, 1> &int64Param) {
  std::ifstream floatParamFile(floatParamPath, std::ios::in | std::ios::binary);
  if (!floatParamFile.is_open()) {
    std::string errMsg = "Failed to open float param file: " +
                         std::filesystem::canonical(floatParamPath).string();
    throw std::runtime_error(errMsg);
  }
  floatParamFile.read(reinterpret_cast<char *>(floatParam.getData()),
                      floatParam.getSize() * sizeof(float));
  if (floatParamFile.fail()) {
    throw std::runtime_error("Failed to read float param file");
  }
  floatParamFile.close();


  std::ifstream int64ParamFile(int64ParamPath, std::ios::in | std::ios::binary);
  if (!int64ParamFile.is_open()) {
    std::string errMsg = "Failed to open int64 param file: " +
                         std::filesystem::canonical(int64ParamPath).string();
    throw std::runtime_error(errMsg);
  }
  int64ParamFile.read(reinterpret_cast<char *>(int64Param.getData()),
                      int64Param.getSize() * sizeof(long long));
  if (int64ParamFile.fail()) {
    throw std::runtime_error("Failed to read int64 param file");
  }
  int64ParamFile.close();
}

// Softmax function.
void softmax(float *input, size_t size) {
  size_t i;
  float max_value = -INFINITY;
  double sum = 0.0;
  // Find the maximum value in the input array for numerical stability.
  for (i = 0; i < size; ++i) {
    if (max_value < input[i]) {
      max_value = input[i];
    }
  }
  // Calculate the sum of the exponentials of the input elements, normalized by
  // the max value.
  for (i = 0; i < size; ++i) {
    sum += exp(input[i] - max_value);
  }
  // Normalize the input array with the softmax calculation.
  for (i = 0; i < size; ++i) {
    input[i] = exp(input[i] - max_value) / sum;
  }
}

std::string getLabel(int idx) {
  std::string mobilenetDir = getenv("MOBILENETV3_EXAMPLE_PATH");
  std::ifstream in(
      mobilenetDir + "Labels.txt");
  assert(in.is_open() && "Could not read the label file.");
  std::string label;
  for (int i = 0; i < idx; ++i)
    std::getline(in, label);
  std::getline(in, label);
  in.close();
  return label;
}

int main() {
  // Print the title of this example.
  const std::string title = "MobileNetV3 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Preprocess the image to match the input requirements of the model.
  cv::Mat image = imagePreprocessing();

  // Define the sizes of the input and output tensors.
  intptr_t sizesInput[4] = {1, 3, 224, 224};
  intptr_t sizesOutput[2] = {1, 1000};

  // Create input and output containers for the image and model output.
  Img<float, 4> input(image, sizesInput, true);
  MemRef<float, 2> output(sizesOutput);

  // Load model parameters from the specified file.
  std::string mobilenetDir = getenv("MOBILENETV3_EXAMPLE_PATH");
  std::string paramsDir = mobilenetDir + "/arg0.data";
  std::string intDir = mobilenetDir + "/arg1.data";
  MemRef<float, 1> paramsContainerf32({ParamsSize});
  MemRef<long long, 1> ParamsContainerInt64({34});
  loadParameters(paramsDir, intDir, paramsContainerf32, ParamsContainerInt64);
  // Call the forward function of the model.
  _mlir_ciface_forward(&output, &paramsContainerf32, &ParamsContainerInt64, &input);
 
  auto out = output.getData();
  softmax(out, 1000);
  // Find the classification and print the result.
  float maxVal = 0;
  float maxIdx = 0;
  for (int i = 0; i < 1001; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }
  std::cout << "Classification Index: " << maxIdx << std::endl;
  std::cout << "Classification: " << getLabel(maxIdx) << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;

  return 0;
}
