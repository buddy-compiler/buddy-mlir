#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <cstring>
#include <iomanip>
#include <typeinfo>

#include <fp16.h>

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

using fp16_t = uint16_t;
using bf16_t = uint16_t;
using half_t = uint16_t;

float fp162float(fp16_t hf);

fp16_t float2fp16(float f);

float bf162float(bf16_t bfloatBits);

bf16_t float2bf16(float f);

half_t float2half(float f);

float half2float(half_t hf);


/// Capture input message.
void getUserInput(std::string &inputStr);

/// Print [Log] label in bold blue format.
void printLogLabel();

/// Print information for each iteration.
void printIterInfo(size_t iterIdx, std::string str, double time);

/// Tokenize input data in the container.
void tokenizeInput(const std::string &vocabFile,
                   buddy::Text<size_t, 2> &inputContainer,
                   size_t maxTokenLength);

struct ModelConfig {
    size_t paramSize;
    size_t maxVocabSize;
    size_t maxTokenLength;
    size_t hiddenSize;
};

struct QuantizedModelConfig {
    size_t float32ParamSize;
    size_t int64ParamSize;
    size_t int8ParamSize;
    size_t maxVocabSize;
    size_t maxTokenLength;
    size_t hiddenSize;
};

void loadModelConfig(const std::string &configFilePath,
                    ModelConfig &config);

void loadQuantizedModelConfig(const std::string &configFilePath,
                    QuantizedModelConfig &config);

/// Load parameters into data container.
template<typename param_type>
void loadParameters(const std::string &paramFilePath,
                    MemRef<param_type, 1> &params, size_t params_size) {
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
  float *param_cache = (float*)malloc(sizeof(float) * params_size);
  paramFile.read(reinterpret_cast<char *>(param_cache),
                 sizeof(float) * params_size);
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
  if constexpr(sizeof(param_type) == 2) {
    printLogLabel();
    std::cout << "Casting float params to half... " << std::endl;
    for (size_t i = 0; i < params_size; i++) {
        params[i] = float2half(param_cache[i]);
    }
    const auto castEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> castTime =
        castEnd - loadEnd;
    printLogLabel();
    std::cout << "Params cast time: " << (double)(castTime.count()) / 1000
              << "s\n"
              << std::endl;
  } else {
    memcpy(params.getData(), param_cache, sizeof(float) * params_size);
  }
  free(param_cache);
  
}