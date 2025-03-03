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
#include <string>
#include <utility>
#include <vector>

using namespace buddy;

// Declare BERT forward function.
extern "C" void
_mlir_ciface_forward(MemRef<float, 2> *result, MemRef<float, 1> *arg0,
                     MemRef<long long, 1> *arg1, MemRef<long long, 2> *arg2,
                     MemRef<long long, 2> *arg3, MemRef<long long, 2> *arg4);

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

int main() {
  /// Print the title of this example.
  const std::string title = "BERT Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;
  
  /// Load weights to MemRef container.
  std::string bertDir = BERT_EXAMPLE_PATH;
  std::string bertBuildDir = BERT_EXAMPLE_BUILD_PATH;
  MemRef<float, 1> arg0({109486854});
  MemRef<long long, 1> arg1({512});
  loadParameters(bertBuildDir + "/arg0.data",
                 bertBuildDir + "/arg1.data", arg0, arg1);

  /// Get user message and build Text container.
  std::cout << "What sentence do you want to say to BERT?" << std::endl;
  std::cout << ">>> ";
  std::string pureStr;
  std::getline(std::cin, pureStr);
  Text<long long, 2> pureStrContainer(pureStr);

  /// Define vacabulary and tokenize the 
  std::string vocabDir = bertDir + "/vocab.txt";
  pureStrContainer.tokenizeBert(vocabDir, 5);

  /// Initialize data containers.
  MemRef<float, 2> result({1, 6});
  MemRef<long long, 2> attention_mask({1, 5}, 1LL);
  MemRef<long long, 2> token_type_ids({1, 5}, 0LL);

  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  /// Execute forward inference of the model.
  _mlir_ciface_forward(&result, &arg0, &arg1, &pureStrContainer,
                       &token_type_ids, &attention_mask);
  
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
  /// Find the selected emotion.
  int predict_label = -1;
  float max_logits = std::numeric_limits<float>::min();
  for (int i = 0; i < 6; i++) {
    if (max_logits < result.getData()[i]) {
      max_logits = result.getData()[i];
      predict_label = i;
    }
  }
  std::vector<std::string> emotion = {"sadness", "joy",  "love",
                                      "anger",   "fear", "surprise"};
  std::cout << "\033[33;1m[Result] \033[0m";
  std::cout << "The emotion of your sentence is ";
  std::cout << "\033[32;1m" << emotion[predict_label] << "\033[0m";
  std::cout << "." << std::endl;

  /// Print the performance.
  std::cout << "\033[33;1m[Time] \033[0m";
  std::cout << inferenceTime.count() << " ms"
            << std::endl;

  return 0;
}
