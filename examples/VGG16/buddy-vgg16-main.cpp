//===- InceptionV3Benchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for e2e inceptionv3.
//
//===----------------------------------------------------------------------===//

#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
constexpr size_t ParamsSize = 44426;

// Declare the inceptionv3 C interface.
extern "C" {
void _mlir_ciface_forward(MemRef<float, 2> *output, MemRef<float, 1> *arg0,Img<float, 4> *input);
}

const cv::Mat imagePreprocessing() {
  std::string vggDir = getenv("VGG16_EXAMPLE_PATH");
  std::string inputImage = vggDir+"dog.png";
  assert(!inputImage.empty() && "Could not read the image.");
  cv::Mat resizedImage;
  int imageWidth = 224;
  int imageHeight = 224;
  cv::resize(inputImage, resizedImage, cv::Size(imageWidth, imageHeight),
             cv::INTER_LINEAR);
  return resizedImage;
}


/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

void loadParameters(const std::string &floatParamPath,
                    MemRef<float, 1> &floatParam
                    ) {
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

}
// Softmax function.
void softmax(float *input, size_t size) {
  assert(0 <= size <= sizeof(input) / sizeof(float));
  int i;
  float m, sum, constant;
  m = -INFINITY;
  for (i = 0; i < size; ++i) {
    if (m < input[i]) {
      m = input[i];
    }
  }

  sum = 0.0;
  for (i = 0; i < size; ++i) {
    sum += exp(input[i] - m);
  }

  constant = m + log(sum);
  for (i = 0; i < size; ++i) {
    input[i] = exp(input[i] - constant);
  }
}


int main(){
  // Print the title of this example.
  const std::string title = "Vgg16 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  cv::Mat image = imagePreprocessing();

  intptr_t sizesInput[4] = {1,1,224,224};
  intptr_t sizesOutnput[2] = {1, 1000};
  std::cout << "imagesize:"<<image.size() << std::endl;
  Img<float, 4> input(image, sizesInput, true);
  MemRef<float, 2> output(sizesOutnput);
  
  std::string vggDir = getenv("VGG16_EXAMPLE_PATH");
  std::string paramsDir = vggDir+"/arg0.data";
  MemRef<float, 1> paramsContainer({ParamsSize});
  loadParameters(paramsDir, paramsContainer);
  
  _mlir_ciface_forward(&output,&paramsContainer,&input);
  
  auto out = output.getData();//结果展平
  softmax(out, 1000);
  std::cout << "out:" << *out <<std::endl;
  // Find the classification and print the result.
  float maxVal = 0;
  float maxIdx = 0;
  for (int i = 0; i < 1001; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }
  std::cout << "Classification: " << maxIdx << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;
  return 0;
}