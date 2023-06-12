//====--- sentiment-classification.cpp - Example of bert e2e model---------===//
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
// This file implements a sentiment classification example with a bert model.
// During the execution of this example, the model will be first compiled into
// an object file before being linked with to produce the executable file.

// The bert model is generated from https://huggingface.co/prajjwal1/bert-tiny
// by using torch-mlir. Specifically, we dump it in the output type of
// LINALG_ON_TENSORS with weights and remove the unused `global_seed` function
// from the mlir file.
//
//===----------------------------------------------------------------------===//

#include "../../frontend/Interfaces/buddy/Core/Container.h"
#include <iostream>
#include <math.h>

using namespace std;

// Declare the bert C interface.
extern "C" {
void _mlir_ciface_forward(MemRef<float, 2> *output,
                          MemRef<long long, 2> *input);
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

// This model allows input of exactly 10 words, plus a header `[UNK]` and
// a trailer `[CLS]`.
intptr_t sizesInput[2] = {1, 12};
intptr_t sizesOutput[2] = {1, 2};

// Refering to https://huggingface.co/prajjwal1/bert-tiny/blob/main/vocab.txt,
// these tokens represent a specific sentence "The quick brown fox jumps over
// the lazy dog."
long long sentenceTokens[12] = {101,  1996, 4248,  2829, 4419, 14523,
                                2058, 1996, 13971, 3899, 1012, 102};

int main(int argc, char *argv[]) {
  // TODO: Find a way to encode any input sentences given by command line
  // arguments into tokens, pad the tokens to the specified length or raise an
  // error when the given sentence is too long.
  MemRef<long long, 2> input(sentenceTokens, sizesInput);
  MemRef<float, 2> output(sizesOutput);

  _mlir_ciface_forward(&output, &input);
  auto out = output.getData();
  softmax(out, 2);
  printf("The probability of positive label: %.2lf\n", out[1]);
  printf("The probability of negative label: %.2lf\n", out[0]);
  return 0;
}
