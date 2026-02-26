//===- buddy-yolo26n-main.cpp ---------------------------------------------===//
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

#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImgContainer.h>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

constexpr size_t ParamsSize = 2617048;
constexpr int InputSize = 640;
constexpr int MaxDetections = 300;
constexpr float PadValue = 114.0f / 255.0f;
constexpr float ScoreThreshold = 0.25f;

struct LetterboxResult {
  MemRef<float, 4> input;
  float scale;
  int padTop;
  int padLeft;
  int originalH;
  int originalW;
};

struct Detection {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int classId;
};

extern "C" void _mlir_ciface_forward(MemRef<float, 3> *output,
                                     MemRef<float, 1> *arg0,
                                     MemRef<float, 4> *arg1);

void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }

  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;

  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();

  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << loadTime.count() / 1000.0 << "s\n"
            << std::endl;
}

std::string getBuildDir() {
#ifdef YOLO26_EXAMPLE_BUILD_PATH
  return YOLO26_EXAMPLE_BUILD_PATH;
#else
  return ".";
#endif
}

LetterboxResult letterboxImage(dip::Image<float, 4> &image) {
  const int originalH = static_cast<int>(image.getSizes()[2]);
  const int originalW = static_cast<int>(image.getSizes()[3]);
  const float scale = std::min(static_cast<float>(InputSize) / originalH,
                               static_cast<float>(InputSize) / originalW);
  const int resizedH =
      std::max(1, static_cast<int>(std::round(originalH * scale)));
  const int resizedW =
      std::max(1, static_cast<int>(std::round(originalW * scale)));
  const int padTop = (InputSize - resizedH) / 2;
  const int padLeft = (InputSize - resizedW) / 2;

  MemRef<float, 4> resized = dip::Resize4D_NCHW(
      &image, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
      std::vector<uint>{1, 3, static_cast<uint>(resizedH),
                        static_cast<uint>(resizedW)});
  MemRef<float, 4> input({1, 3, InputSize, InputSize}, PadValue);

  float *dst = input.getData();
  float *src = resized.getData();
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < resizedH; ++y) {
      const size_t dstOffset =
          (static_cast<size_t>(c) * InputSize + (padTop + y)) * InputSize +
          padLeft;
      const size_t srcOffset =
          (static_cast<size_t>(c) * resizedH + y) * resizedW;
      std::memcpy(dst + dstOffset, src + srcOffset, sizeof(float) * resizedW);
    }
  }

  return {std::move(input), scale, padTop, padLeft, originalH, originalW};
}

std::vector<Detection> postprocess(MemRef<float, 3> &output) {
  const float *out = output.getData();
  const intptr_t *strides = output.getStrides();
  std::vector<Detection> candidates;
  const int detCount = static_cast<int>(output.getSizes()[1]);
  candidates.reserve(detCount);
  for (int i = 0; i < detCount; ++i) {
    const intptr_t base = i * strides[1];
    const float score = out[base + 4 * strides[2]];
    if (score >= ScoreThreshold) {
      candidates.push_back(
          {out[base + 0 * strides[2]], out[base + 1 * strides[2]],
           out[base + 2 * strides[2]], out[base + 3 * strides[2]], score,
           static_cast<int>(out[base + 5 * strides[2]])});
    }
  }
  return candidates;
}

Detection mapToOriginalImage(const Detection &det,
                             const LetterboxResult &letterbox) {
  const float invScale = 1.0f / letterbox.scale;
  Detection mapped = det;
  mapped.x1 = (det.x1 - letterbox.padLeft) * invScale;
  mapped.y1 = (det.y1 - letterbox.padTop) * invScale;
  mapped.x2 = (det.x2 - letterbox.padLeft) * invScale;
  mapped.y2 = (det.y2 - letterbox.padTop) * invScale;
  mapped.x1 =
      std::clamp(mapped.x1, 0.0f, static_cast<float>(letterbox.originalW - 1));
  mapped.y1 =
      std::clamp(mapped.y1, 0.0f, static_cast<float>(letterbox.originalH - 1));
  mapped.x2 =
      std::clamp(mapped.x2, 0.0f, static_cast<float>(letterbox.originalW - 1));
  mapped.y2 =
      std::clamp(mapped.y2, 0.0f, static_cast<float>(letterbox.originalH - 1));
  return mapped;
}

int main(int argc, char **argv) {
  const std::string title = "YOLO26n Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <image_path(.bmp preferred)> [arg0.data path]" << std::endl;
    return 1;
  }

  const std::string imagePath = argv[1];
  const std::string paramsPath =
      argc >= 3 ? argv[2] : getBuildDir() + "/arg0.data";
  std::string imageExt = std::filesystem::path(imagePath).extension().string();
  std::transform(imageExt.begin(), imageExt.end(), imageExt.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (imageExt != ".bmp") {
    std::cerr << "Only .bmp image is supported in this example." << std::endl;
    return 1;
  }

  dip::Image<float, 4> image(imagePath, dip::DIP_RGB, true /* norm */);
  LetterboxResult letterbox = letterboxImage(image);

  MemRef<float, 1> params({ParamsSize});
  loadParameters(paramsPath, params);

  MemRef<float, 3> output({1, MaxDetections, 6});
  const auto inferStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward(&output, &params, &letterbox.input);
  const auto inferEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferTime =
      inferEnd - inferStart;
  printLogLabel();
  std::cout << "Inference time: " << inferTime.count() / 1000.0 << "s"
            << std::endl;
  const intptr_t *outSizes = output.getSizes();
  const intptr_t *outStrides = output.getStrides();
  printLogLabel();
  std::cout << "Output shape/stride: [" << outSizes[0] << ", " << outSizes[1]
            << ", " << outSizes[2] << "] / [" << outStrides[0] << ", "
            << outStrides[1] << ", " << outStrides[2] << "]" << std::endl;

  const std::vector<Detection> detections = postprocess(output);
  int validCount = 0;
  std::cout << std::fixed << std::setprecision(4);
  for (const Detection &det : detections) {
    if (det.score < ScoreThreshold) {
      continue;
    }
    const Detection mapped = mapToOriginalImage(det, letterbox);
    std::cout << "[" << validCount << "] class_id=" << mapped.classId
              << " score=" << mapped.score << " box=(" << mapped.x1 << ", "
              << mapped.y1 << ", " << mapped.x2 << ", " << mapped.y2 << ")"
              << std::endl;
    ++validCount;
  }
  std::cout << "Detections: " << validCount << std::endl;

  return 0;
}
