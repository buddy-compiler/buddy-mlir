//===- buddy-vgg-main.cpp -----------------------------------------------===//
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

constexpr size_t ParamsSize = 138357544;
const std::string ImgName = "YuTu.png";

/// Declare vgg forward function.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *arg0,
                                     Img<float, 4> *input);

/// Function for preprocessing the image to match model input requirements.
const cv::Mat imagePreprocessing() {
  // Get the directory of the vgg example and construct the image path.
  std::string vggDir = getenv("VGG_EXAMPLE_PATH");
  std::string imgPath = vggDir + "/images/" + ImgName;
  // Read the image in grayscale mode.
  cv::Mat inputImage = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
  assert(!inputImage.empty() && "Could not read the image.");
  cv::Mat resizedImage;
  int imageWidth = 224;
  int imageHeight = 224;
  // Resize the image to 28x28 pixels.
  cv::resize(inputImage, resizedImage, cv::Size(imageWidth, imageHeight),
             cv::INTER_LINEAR);
  return resizedImage;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  // Open the parameter file in binary mode.
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  // Print the canonical path of the parameter file.
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  // Read the parameter data into the provided memory reference.
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * (params.getSize()));
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;
}

/// Softmax function to convert logits to probabilities.
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

int main() {
  // Print the title of this example.
  const std::string title = "VGG Inference Powered by Buddy Compiler";
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
  std::string vggDir = getenv("VGG_EXAMPLE_PATH");
  std::string paramsDir = vggDir + "/arg0.data";
  MemRef<float, 1> paramsContainer({ParamsSize});
  loadParameters(paramsDir, paramsContainer);

  // Call the forward function of the model.
  _mlir_ciface_forward(&output, &paramsContainer, &input);

  // Apply softmax to the output logits to get probabilities.
  auto out = output.getData();
  softmax(out, 1000);

  // Find the classification and print the result.
  float maxVal = 0;
  float maxIdx = 0;
  for (int i = 0; i < 1000; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }

  std::cout << "Classification: " << maxIdx << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;

  return 0;
}
