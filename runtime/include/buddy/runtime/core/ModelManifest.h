//===- ModelManifest.h - Read .rax manifest and resolve paths -------------===//
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
// The .rax file produced by:
//   rax-pack model.mlir -o model.rax
//
// encodes:
//   constants[0]             → External weights URI  (e.g. "file:arg0.data")
//   code_objects[0]          → HostSharedLib URI      (e.g. "file:model.so")
//   module_attrs["vocab_uri"]→ vocab file URI         (e.g. "file:vocab.txt")
//   module_attrs["model_name"]→ model identifier     (e.g. "deepseek_r1_fp32")
//
// All "file:X" URIs are resolved relative to the .rax file's directory,
// yielding absolute paths ready to pass to fopen / dlopen.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_MODELMANIFEST_H
#define BUDDY_RUNTIME_CORE_MODELMANIFEST_H

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// FlatBuffers + generated RAX schema (provided by buddy_rax INTERFACE target)
#include "buddy/runtime/rax/RAX.h"
#include "flatbuffers/flatbuffers.h"

namespace buddy {
namespace runtime {

struct ModelManifest {
  // from module_attrs["model_name"] (e.g. "deepseek_r1_fp32")
  std::string modelName;
  // absolute path to the kernel .so  (dlopen)
  std::string soPath;
  // absolute path to the weight blob  (mmap / fread)
  std::string weightsPath;
  // absolute path to the vocab file   (tokenizer, may be empty)
  std::string vocabPath;

  // Load and resolve from a .rax manifest file.
  // Throws std::runtime_error on any parse / missing-field error.
  static ModelManifest loadFromRax(const std::string &raxPath) {
    namespace fs = std::filesystem;

    // --- Read raw bytes ---------------------------------------------------
    std::ifstream ifs(raxPath, std::ios::binary);
    if (!ifs)
      throw std::runtime_error("ModelManifest: cannot open " + raxPath);
    ifs.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(n);
    if (n && !ifs.read(reinterpret_cast<char *>(bytes.data()), n))
      throw std::runtime_error("ModelManifest: read failed: " + raxPath);

    // --- Verify FlatBuffer identifier ------------------------------------
    if (bytes.size() < 8 ||
        !flatbuffers::BufferHasIdentifier(bytes.data(), "RAX0"))
      throw std::runtime_error("ModelManifest: not a valid RAX0 file: " +
                               raxPath);

    flatbuffers::Verifier ver(bytes.data(), bytes.size());
    if (!rhal::rax::VerifyModuleBuffer(ver))
      throw std::runtime_error("ModelManifest: RAX verification failed: " +
                               raxPath);

    // --- Parse module -----------------------------------------------------
    const rhal::rax::Module *mod = rhal::rax::GetModule(bytes.data());

    fs::path baseDir = fs::absolute(fs::path(raxPath)).parent_path();

    auto resolveUri = [&](const flatbuffers::String *uri,
                          const char *field) -> std::string {
      if (!uri || uri->size() == 0)
        throw std::runtime_error(
            std::string("ModelManifest: missing URI for ") + field);
      std::string s = uri->str();
      if (s.rfind("file:", 0) == 0)
        return (baseDir / fs::path(s.substr(5))).string();
      return s; // absolute path or passthrough
    };

    ModelManifest out;

    // --- Code object → soPath (first HostSharedLib entry) ----------------
    if (!mod->code_objects() || mod->code_objects()->size() == 0)
      throw std::runtime_error("ModelManifest: no code_objects in " + raxPath);
    for (auto co : *mod->code_objects()) {
      if (co->kind() == rhal::rax::CodeObjectKind_HostSharedLib) {
        out.soPath = resolveUri(co->uri(), "code_object.uri");
        break;
      }
    }
    if (out.soPath.empty())
      throw std::runtime_error(
          "ModelManifest: no HostSharedLib code object in " + raxPath);

    // --- Constant → weightsPath (first External constant) ----------------
    if (!mod->constants() || mod->constants()->size() == 0)
      throw std::runtime_error("ModelManifest: no constants in " + raxPath);
    for (auto c : *mod->constants()) {
      if (c->storage() == rhal::rax::ConstantStorage_External) {
        out.weightsPath = resolveUri(c->uri(), "constant.uri");
        break;
      }
    }
    if (out.weightsPath.empty())
      throw std::runtime_error(
          "ModelManifest: no External constant (weights) in " + raxPath);

    // --- Module attrs: vocab_uri, model_name (optional) ------------------
    if (mod->attrs()) {
      for (auto kv : *mod->attrs()) {
        if (!kv->key())
          continue;
        std::string key = kv->key()->str();
        if (key == "vocab_uri" && kv->value() && kv->value()->size() > 0)
          out.vocabPath = resolveUri(kv->value(), "vocab_uri");
        else if (key == "model_name" && kv->value() && kv->value()->size() > 0)
          out.modelName = kv->value()->str();
      }
    }

    return out;
  }
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_MODELMANIFEST_H
