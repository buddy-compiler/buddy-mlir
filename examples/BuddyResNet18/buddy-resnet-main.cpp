//===- buddy-resnet-main.cpp ----------------------------------------------===//
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
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImgContainer.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

constexpr size_t ParamsSize = 11699112;
const std::string ImgName = "dog-224*224.png";

// Declare the resnet C interface.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *arg0,
                                     MemRef<float, 4> *input);

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
  std::string resnetDir = RESNET_EXAMPLE_PATH;
  std::ifstream in(resnetDir + "/Labels.txt");
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
  const std::string title = "ResNet Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  // Define the sizes of the input and output tensors.
  intptr_t sizesOutput[2] = {1, 1000};

  // Create input and output containers for the image and model output.
  std::string resnetDir = RESNET_EXAMPLE_PATH;
  std::string resnetBuildDir = RESNET_EXAMPLE_BUILD_PATH;
  std::string imgPath = resnetDir + "/images/" + ImgName;
  dip::Image<float, 4> input(imgPath, dip::DIP_RGB, true /* norm */);
  MemRef<float, 4> inputResize = dip::Resize4D_NCHW(
      &input, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
      {1, 3, 224, 224} /*{image_cols, image_rows}*/);

  MemRef<float, 2> output(sizesOutput);

  // Load model parameters from the specified file.
  std::string paramsDir = resnetBuildDir + "/arg0.data";
  MemRef<float, 1> paramsContainer({ParamsSize});
  loadParameters(paramsDir, paramsContainer);

  const auto inferStart = std::chrono::high_resolution_clock::now();
  // Call the forward function of the model.
  _mlir_ciface_forward(&output, &paramsContainer, &inputResize);
  const auto inferEnd = std::chrono::high_resolution_clock::now();

  const std::chrono::duration<double, std::milli> inferTime =
      inferEnd - inferStart;

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

  printLogLabel();
  std::cout << "Inference time: " << inferTime.count() / 1000 << std::endl;

  return 0;
}
