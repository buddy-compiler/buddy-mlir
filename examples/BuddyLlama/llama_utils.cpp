#include "llama_utils.h"
#include <cassert>

using namespace buddy;

// ----------------- FP16/BP16 <-> F32 conversion utils & builtins ----------------------

float fp162float(fp16_t hf) {
    return fp16_ieee_to_fp32_value(hf);
}

fp16_t float2fp16(float f) {
  return fp16_ieee_from_fp32_value(f);
}

// Converts the 16 bit representation of a bfloat value to a float value. This
// implementation is adapted from Eigen.
union Float32Bits {
  uint32_t u;
  float f;
};
float bf162float(bf16_t bfloatBits) {
  Float32Bits floatBits;
  floatBits.u = static_cast<uint32_t>(bfloatBits) << 16;
  return floatBits.f;
}

bf16_t float2bf16(float f) {
  Float32Bits floatBits;
  floatBits.f = f;
  return static_cast<bf16_t>(floatBits.u >> 16);
}

half_t float2half(float f) {
#if defined(LLAMA_FP16_TYPE)
  return float2fp16(f);
#elif defined(LLAMA_BF16_TYPE)
  return float2bf16(f);
#endif
  assert(false);
}

float half2float(half_t hf) {
#if defined(LLAMA_FP16_TYPE)
  return fp162float(hf);
#elif defined(LLAMA_BF16_TYPE)
  return bf162float(hf);
#endif
  assert(false);
}

/// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Print information for each iteration.
void printIterInfo(size_t iterIdx, std::string str, double time) {
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

/// Tokenize input data in the container.
void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer,
                   size_t maxTokenLength) {
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
            << std::endl;
  const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeLlama(vocabFile, maxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}


void loadModelConfig(const std::string &configFilePath, ModelConfig &config) {
  std::cout << "Config file: " << std::filesystem::canonical(configFilePath)
            << std::endl;
  std::ifstream configFile(configFilePath, std::ios::in);
  std::string s;
  configFile >> s >> config.paramSize;
  assert(s == "paramSize");
  std::cout << s << " = " << config.paramSize << std::endl;
  configFile >> s >> config.hiddenSize;
  assert(s == "hiddenSize");
  std::cout << s << " = " << config.hiddenSize << std::endl;
  configFile >> s >> config.maxVocabSize;
  assert(s == "maxVocabSize");
  std::cout << s << " = " << config.maxVocabSize << std::endl;
  configFile >> s >> config.maxTokenLength;
  assert(s == "maxTokenLength");
  std::cout << s << " = " << config.maxTokenLength << std::endl;
}