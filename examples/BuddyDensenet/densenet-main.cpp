//===- bert-main.cpp ------------------------------------------------------===//
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
#include <buddy/LLM/TextContainer.h>
#include <filesystem>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

using namespace buddy;

// Declare DenseNet forward function.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *result,
                                     MemRef<float, 1> *arg0,
                                     MemRef<long long, 1> *arg1,
                                     MemRef<float, 4> *arg2);

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

int main(int argc, char **argv) {
  /// Print the title of this example.
  const std::string title = "DenseNet Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Load weights to MemRef container.
  MemRef<float, 1> arg0({8062504});
  MemRef<long long, 1> arg1({121});
  loadParameters("../../examples/BuddyBert/arg0.data",
                 "../../examples/BuddyBert/arg1.data", arg0, arg1);

  if (argc != 2) {
    std::cout << "Need Img Path" << std::endl;
  }
  /// Get user image.
  std::cout << "Read Img:" << argv[1] << std::endl;
  cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(224, 224));

  MemRef<float, 4> input({1, 3, 224, 224});
  float *dst = input.getData();
  unsigned char *src = resize_image.ptr<unsigned char>();
  // from BGR to channal RGB
  for (int i = 0; i < 224; ++i) {
    for (int j = 0; j < 224; ++j) {
      float r = src[(i * 224 + j) * 3 + 2], g = src[(i * 224 + j) * 3 + 1],
            b = src[(i * 224 + j) * 3];

      dst[i * 224 + j] = (r / 255 - 0.485) / 0.229;
      dst[224 * 224 + i * 224 + j] = (g / 255 - 0.456) / 0.224;
      dst[224 * 224 * 2 + i * 224 + j] = (b / 255 - 0.406) / 0.225;
    }
  }

  /// Initialize data containers.
  MemRef<float, 2> result({1, 1000});

  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  /// Execute forward inference of the model.
  _mlir_ciface_forward(&result, &arg0, &arg1, &input);

  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;
  /// Find the selected emotion.
  int predict_label = -1;
  float max_logits = std::numeric_limits<float>::min();
  for (int i = 0; i < 1000; i++) {
    if (max_logits < result.getData()[i]) {
      max_logits = result.getData()[i];
      predict_label = i;
    }
  }

  std::cout << "\033[33;1m[Result] \033[0m";
  std::cout << "The label of your image is ";
  std::cout << "\033[32;1m" << predict_label << "\033[0m";
  std::cout << "." << std::endl;

  /// Print the performance.
  std::cout << "\033[33;1m[Time] \033[0m";
  std::cout << inferenceTime.count() << " ms" << std::endl;

  return 0;
}
