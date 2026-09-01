//===- BgeM3EmbeddingModelPlugin.cpp - BGE-M3 embedding plugin ABI --------===//
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

#include "buddy/runtime/models/BgeM3EmbeddingModel.h"

extern "C" buddy::runtime::EmbeddingModel *buddy_create_embedding_model_v1() {
  return new buddy::runtime::BgeM3EmbeddingModel();
}

extern "C" void
buddy_destroy_embedding_model_v1(buddy::runtime::EmbeddingModel *model) {
  delete model;
}

extern "C" const char *buddy_embedding_model_type_v1() { return "bge_m3"; }
