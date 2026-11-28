// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Helper for rax-kernel-metadata.mlir; does not require Python or Triton.

#include "buddy/runtime/core/ExternalKernel.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxDependenciesHandle.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace fs = std::filesystem;
using buddy::runtime::ModelManifest;

static void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

static void writeInvalidMetadata(const fs::path &path, bool duplicate) {
  flatbuffers::FlatBufferBuilder b;
  std::vector<flatbuffers::Offset<rhal::rax::KV>> attrs;
  attrs.push_back(
      rhal::rax::CreateKV(b, b.CreateString("optional"),
                          b.CreateString(duplicate ? "true" : "yes")));
  if (duplicate)
    attrs.push_back(rhal::rax::CreateKV(b, b.CreateString("optional"),
                                        b.CreateString("false")));
  auto code = rhal::rax::CreateCodeObject(
      b, 1, b.CreateString("optional"), rhal::rax::CodeObjectKind_HostSharedLib,
      0, b.CreateString("file:optional.so"), 0, 0, b.CreateVector(attrs));
  auto codes = b.CreateVector(&code, 1);
  auto module =
      rhal::rax::CreateModule(b, b.CreateString("RAX"), 0,
                              rhal::rax::Endianness_Little, 0, 0, 0, 0, codes);
  b.Finish(module, "RAX0");
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char *>(b.GetBufferPointer()),
               b.GetSize());
  require(bool(stream), "write invalid-metadata fixture failed");
}

static void prepare(const fs::path &dir) {
  fs::create_directories(dir);
  for (const char *name : {"primary.so", "dependency.so", "optional.so"})
    fs::copy_file(BUDDY_RAX_METADATA_TEST_PLUGIN, dir / name,
                  fs::copy_options::overwrite_existing);
  writeInvalidMetadata(dir / "invalid-runtime.rax", false);
  writeInvalidMetadata(dir / "duplicate-runtime.rax", true);
  std::cout << "Prepared RAX metadata fixtures\n";
}

static void expectLoadError(const fs::path &path, const char *expected) {
  try {
    ModelManifest::loadFromRax(path.string());
  } catch (const std::runtime_error &error) {
    require(std::string(error.what()).find(expected) != std::string::npos,
            "unexpected manifest error");
    return;
  }
  throw std::runtime_error("invalid metadata was accepted");
}

static int32_t fallback(const BuddyKernelCallV1 *call) {
  auto *output = static_cast<float *>(call->output.data);
  for (int i = 0; i < 8; ++i)
    output[i] = 7.f;
  return BUDDY_KERNEL_OK;
}

static void check(const fs::path &path, const std::string &mode) {
  auto manifest = ModelManifest::loadFromRax(path.string());
  std::ifstream stream(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), {});
  const auto *module = rhal::rax::GetModule(bytes.data());
  require(module->buffers() && module->buffers()->size() == 2,
          "buffer table lost");
  auto *legacy = module->buffers()->Get(1)->type();
  require(legacy->layout() == rhal::rax::Layout_Any && !legacy->strides(),
          "legacy buffer defaults changed");

  if (mode == "legacy") {
    require(manifest.codeObjects.size() == 2, "legacy code objects changed");
    require(manifest.functions.empty(), "legacy functions changed");
    for (const auto &code : manifest.codeObjects)
      require(!code.optional && code.entrySymbol.empty() && code.attrs.empty(),
              "legacy code object defaults changed");
    require(manifest.soPath == manifest.codeObjects[0].path &&
                manifest.dependentSoPaths ==
                    std::vector<std::string>{manifest.codeObjects[1].path},
            "legacy dependency ordering changed");
    std::cout << "PASS legacy RAX defaults\n";
    return;
  }

  require(manifest.codeObjects.size() == 3, "code object table lost");
  const auto &optional = manifest.codeObjects[0];
  require(optional.id == 2 && optional.name == "triton_softmax" &&
              optional.backend == "triton_cpu" && optional.optional &&
              optional.entrySymbol == "buddy_external_kernel_query_v1",
          "optional kernel identity or entry lost");
  const std::unordered_map<std::string, std::string> expected{
      {"kernel_abi", "buddy.external.v1"},
      {"operation", "softmax"},
      {"dtype", "f32"},
      {"layout", "row_major"},
      {"shape", "2,4"},
      {"strides", "4,1"},
      {"axis", "1"},
      {"target", "test-host"},
      {"required_isa", "test-feature"},
      {"optional", "true"}};
  require(optional.attrs == expected, "kernel attributes did not round trip");
  require(manifest.buffers.size() == 2 &&
              manifest.buffers[0].dtype == rhal::rax::DType_F32 &&
              manifest.buffers[0].layout == rhal::rax::Layout_RowMajor &&
              manifest.buffers[0].shape == std::vector<int64_t>{2, 4} &&
              manifest.buffers[0].strides == std::vector<int64_t>{4, 1} &&
              manifest.buffers[1].layout == rhal::rax::Layout_Any &&
              manifest.buffers[1].strides.empty(),
          "resolved buffer metadata did not round trip");
  require(!manifest.codeObjects[1].optional &&
              !manifest.codeObjects[2].optional,
          "mandatory code object became optional");
  require(manifest.soPath == manifest.codeObjects[1].path &&
              manifest.dependentSoPaths ==
                  std::vector<std::string>{manifest.codeObjects[2].path},
          "optional kernel leaked into mandatory dependencies or primary code");

  // Function dispatch metadata: buffers and scalars bound in ABI order.
  require(manifest.functions.size() == 1, "function table lost");
  const auto &function = manifest.functions[0];
  require(function.name == "softmax" &&
              function.inputs == std::vector<uint32_t>{1} &&
              function.outputs == std::vector<uint32_t>{2} &&
              function.dispatches.size() == 1,
          "function dispatch metadata did not round trip");
  const auto &dispatch = function.dispatches[0];
  require(dispatch.codeObjectId == 2 &&
              dispatch.bufferArgs == std::vector<uint32_t>{1, 2} &&
              dispatch.scalars == std::vector<int64_t>{-1} &&
              dispatch.grid == std::vector<uint32_t>{1, 1, 1} &&
              dispatch.block == std::vector<uint32_t>{1, 1, 1},
          "dispatch args/scalars/dims did not round trip");

  auto *type = module->buffers()->Get(0)->type();
  require(type->dtype() == rhal::rax::DType_F32 &&
              type->layout() == rhal::rax::Layout_RowMajor &&
              type->shape()->dims()->size() == 2 &&
              type->shape()->dims()->Get(0) == 2 &&
              type->shape()->dims()->Get(1) == 4 && type->strides() &&
              type->strides()->size() == 2 && type->strides()->Get(0) == 4 &&
              type->strides()->Get(1) == 1,
          "tensor dtype/shape/layout/strides did not round trip");
  if (mode == "embedded") {
    require(optional.uri == "payload:optional.so" &&
                fs::is_regular_file(optional.path),
            "embedded kernel was not resolved/extracted");
  } else {
    require(optional.uri == "file:optional.so" && !fs::exists(optional.path),
            "missing optional-file case was not exercised");
  }

  // Exercise the same mandatory dependency view used before runner creation.
  // The missing optional library must not abort this step.
  buddy::runtime::RaxDependenciesHandle dependencies(
      {manifest.soPath, manifest.dependentSoPaths[0]}, "metadata-test");
  // Plain mode has no library; embedded mode has a fixture with no query.
  // In both cases the optional kernel is unavailable, so dispatch can recover.
  buddy::runtime::ExternalKernel kernel(optional.path);
  require(!kernel.diagnostic().empty(), "unavailable kernel was accepted");
  float input[8] = {}, output[8] = {};
  int64_t shape[] = {2, 4}, strides[] = {4, 1}, axis = -1;
  BuddyKernelCallV1 call{BUDDY_KERNEL_ABI_V1,
                         sizeof(BuddyKernelCallV1),
                         BUDDY_KERNEL_SOFTMAX,
                         0,
                         {input, sizeof(input), 2, BUDDY_KERNEL_F32,
                          BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                         {output, sizeof(output), 2, BUDDY_KERNEL_F32,
                          BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                         &axis,
                         1,
                         nullptr,
                         0};
  auto result = kernel.dispatch(call, fallback);
  require(result.status == BUDDY_KERNEL_OK &&
              result.path == buddy::runtime::KernelPath::Fallback,
          "unavailable optional kernel did not fall back");
  for (float value : output)
    require(value == 7.f, "fallback output was not returned");
  std::cout
      << "PASS " << mode
      << " RAX metadata round trip; mandatory dependencies=1; path=fallback\n";
}

int main(int argc, char **argv) {
  try {
    require(
        argc == 3,
        "usage: metadata-check <prepare|plain|embedded|legacy|invalid> <path>");
    std::string mode = argv[1];
    if (mode == "prepare")
      prepare(argv[2]);
    else if (mode == "invalid") {
      fs::path dir = argv[2];
      expectLoadError(dir / "invalid-runtime.rax", "optional must be");
      expectLoadError(dir / "duplicate-runtime.rax",
                      "duplicate code object metadata");
      std::cout << "PASS malformed runtime metadata rejected\n";
    } else {
      require(mode == "plain" || mode == "embedded" || mode == "legacy",
              "unknown mode");
      check(argv[2], mode);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
