#include <buddy/Core/Container.h>
#include <buddy/DIP/ImgContainer.h>
#include <buddy/DIP/DIP.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

constexpr size_t ParamsSize = 11191242;
const std::string ImgName = "plane.bmp";

/// Declare ResNet forward function.
extern "C" void _mlir_ciface_forward(MemRef<float, 2> *output,
                                     MemRef<float, 1> *arg0,
                                     MemRef<float, 4> *input);

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Load parameters into data container.
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



    // 输出读取的前二十个参数
  std::cout << "[Debug] First 20 parameters from .data file:" << std::endl;
  for (int i = 9536; i < 9600 && i < params.getSize(); ++i) {
    std::cout << "Parameter[" << i << "] = " << params.getData()[i] << std::endl;
  }

}


/// Softmax function to convert logits to probabilities.
void softmax(float *input, size_t size) {
  size_t i;
  float max_value = -INFINITY;
  double sum = 0.0;
  for (i = 0; i < size; ++i) {
    if (max_value < input[i]) {
      max_value = input[i];
    }
  }
  for (i = 0; i < size; ++i) {
    sum += std::exp(input[i] - max_value);
  }
  for (i = 0; i < size; ++i) {
    input[i] = std::exp(input[i] - max_value) / sum;
  }
}

int main() {
  const std::string title = "ResNet Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  intptr_t sizesOutput[2] = {1, 10};
  std::string imgPath = "plane.bmp";

  // intptr_t sizesInput[4] = {1, 3, 224, 224};

  // cv::Mat inputImage = cv::imread(imgPath, cv::IMREAD_COLOR);
  // assert(!inputImage.empty() && "Could not read the image.");
  // Img<float, 4> input(inputImage, sizesInput, true);




  // // Read image in RGB mode, normalized
  dip::Image<float, 4> input(imgPath, dip::DIP_RGB, true /* norm */);

  MemRef<float, 4> inputResize = dip::Resize4D_NCHW(
      &input, dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
      {1, 3, 224, 224} /*{image_cols, image_rows}*/);


  MemRef<float, 2> output(sizesOutput);

  std::string paramsDir = "arg0_resnet18.data";
  MemRef<float, 1> paramsContainer({ParamsSize});
  loadParameters(paramsDir, paramsContainer);

  // Call the forward function of the model.
  _mlir_ciface_forward(&output, &paramsContainer, &inputResize);
  auto out = output.getData();


  softmax(out, 10);

  std::cout << "[Debug] Raw output logits: ";
  for (int i =0; i < 10; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << std::endl;

  // Find the classification and print the result.
  float maxVal = 0;
  float maxIdx = 0;
  for (int i = 0; i < 10; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }

  std::cout << "Classification: " << maxIdx << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;

  return 0;
}


