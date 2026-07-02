//===- buddy-deepseek-r1-gpu-main.cpp -------------------------------------===//
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
// DeepSeekR1 GPU inference entry point.
//
// The subgraph MLIR files are compiled with the NVVM (GPU) pipeline so that
// linalg ops are lowered to CUDA kernels via the parallel-loops → GPU path.
// The forward wrappers are compiled with the standard LLVM lowering pipeline.
//
// At runtime the MLIR CUDA runtime (mlir_cuda_runtime) manages GPU memory via
// cudaMallocManaged (unified memory), so outputs returned by the GPU subgraph
// are CPU-accessible without an explicit device-to-host copy.
//
// std::_Exit(0) is used at the end to skip MLIR CUDA runtime atexit handlers
// that attempt to unload modules after the CUDA context is already destroyed.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/time.h>

using namespace buddy;

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, void *stream,
                              bool isHostShared);

// MemRef backed by cudaMallocManaged (unified memory).
// GPU kernels can access this directly without explicit H2D copies.
// Sets `allocated = nullptr` so the base ~MemRef() does not call free().
class CudaManagedMemRef1D : public MemRef<float, 1> {
  float *cuda_ptr_ = nullptr;

public:
  explicit CudaManagedMemRef1D(size_t count) {
    cudaError_t err = cudaMallocManaged(&cuda_ptr_, sizeof(float) * count);
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaMallocManaged failed: ") +
                               cudaGetErrorString(err));
    this->aligned = cuda_ptr_;
    this->allocated = this->aligned; // must match aligned for MLIR descriptor
    this->sizes[0] = static_cast<intptr_t>(count);
    this->strides[0] = 1;
    this->offset = 0;
  }
  ~CudaManagedMemRef1D() {
    if (cuda_ptr_)
      cudaFree(cuda_ptr_);
  }
};

// Generic N-dimensional managed-memory wrapper.
// Accepts the same initializer_list<intptr_t> shape + init-value interface as
// MemRef constructors so existing call sites can be changed with minimal churn.
template <typename T, size_t N> class CudaManagedMemRef : public MemRef<T, N> {
  T *cuda_ptr_ = nullptr;

public:
  explicit CudaManagedMemRef(std::initializer_list<intptr_t> shape,
                             T initVal = T{}) {
    size_t count = 1;
    size_t i = 0;
    for (auto s : shape) {
      this->sizes[i++] = s;
      count *= static_cast<size_t>(s);
    }
    cudaError_t err = cudaMallocManaged(&cuda_ptr_, sizeof(T) * count);
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaMallocManaged failed: ") +
                               cudaGetErrorString(err));
    std::fill(cuda_ptr_, cuda_ptr_ + count, initVal);
    this->aligned = cuda_ptr_;
    this->allocated = this->aligned; // must match aligned for MLIR descriptor
    this->offset = 0;
    this->strides[N - 1] = 1;
    for (int j = static_cast<int>(N) - 2; j >= 0; --j)
      this->strides[j] = this->strides[j + 1] * this->sizes[j + 1];
  }
  ~CudaManagedMemRef() {
    if (cuda_ptr_)
      cudaFree(cuda_ptr_);
  }
};

double total_time = 0;
constexpr size_t ParamsSize = 1777088064;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;

extern "C" double _mlir_ciface_rtclock() {
#ifndef _WIN32
  struct timeval tp;
  int stat = gettimeofday(&tp, nullptr);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
  fprintf(stderr, "Timing utility not implemented on Windows\n");
  return 0.0;
#endif // _WIN32
}

struct MemRefContainer {

  MemRef<float, 4> kv0;
  MemRef<float, 4> kv1;
  MemRef<float, 4> kv2;
  MemRef<float, 4> kv3;
  MemRef<float, 4> kv4;
  MemRef<float, 4> kv5;
  MemRef<float, 4> kv6;
  MemRef<float, 4> kv7;
  MemRef<float, 4> kv8;
  MemRef<float, 4> kv9;
  MemRef<float, 4> kv10;
  MemRef<float, 4> kv11;
  MemRef<float, 4> kv12;
  MemRef<float, 4> kv13;
  MemRef<float, 4> kv14;
  MemRef<float, 4> kv15;
  MemRef<float, 4> kv16;
  MemRef<float, 4> kv17;
  MemRef<float, 4> kv18;
  MemRef<float, 4> kv19;
  MemRef<float, 4> kv20;
  MemRef<float, 4> kv21;
  MemRef<float, 4> kv22;
  MemRef<float, 4> kv23;
  MemRef<float, 4> kv24;
  MemRef<float, 4> kv25;
  MemRef<float, 4> kv26;
  MemRef<float, 4> kv27;
  MemRef<float, 4> kv28;
  MemRef<float, 4> kv29;
  MemRef<float, 4> kv30;
  MemRef<float, 4> kv31;
  MemRef<float, 4> kv32;
  MemRef<float, 4> kv33;
  MemRef<float, 4> kv34;
  MemRef<float, 4> kv35;
  MemRef<float, 4> kv36;
  MemRef<float, 4> kv37;
  MemRef<float, 4> kv38;
  MemRef<float, 4> kv39;
  MemRef<float, 4> kv40;
  MemRef<float, 4> kv41;
  MemRef<float, 4> kv42;
  MemRef<float, 4> kv43;
  MemRef<float, 4> kv44;
  MemRef<float, 4> kv45;
  MemRef<float, 4> kv46;
  MemRef<float, 4> kv47;
  MemRef<float, 4> kv48;
  MemRef<float, 4> kv49;
  MemRef<float, 4> kv50;
  MemRef<float, 4> kv51;
  MemRef<float, 4> kv52;
  MemRef<float, 4> kv53;
  MemRef<float, 4> kv54;
  MemRef<float, 4> kv55;

  MemRef<float, 3> logits;

  std::array<MemRef<float, 4> *, 56> kv_ptrs;

  MemRefContainer(
      MemRef<float, 4> k0, MemRef<float, 4> k1, MemRef<float, 4> k2,
      MemRef<float, 4> k3, MemRef<float, 4> k4, MemRef<float, 4> k5,
      MemRef<float, 4> k6, MemRef<float, 4> k7, MemRef<float, 4> k8,
      MemRef<float, 4> k9, MemRef<float, 4> k10, MemRef<float, 4> k11,
      MemRef<float, 4> k12, MemRef<float, 4> k13, MemRef<float, 4> k14,
      MemRef<float, 4> k15, MemRef<float, 4> k16, MemRef<float, 4> k17,
      MemRef<float, 4> k18, MemRef<float, 4> k19, MemRef<float, 4> k20,
      MemRef<float, 4> k21, MemRef<float, 4> k22, MemRef<float, 4> k23,
      MemRef<float, 4> k24, MemRef<float, 4> k25, MemRef<float, 4> k26,
      MemRef<float, 4> k27, MemRef<float, 4> k28, MemRef<float, 4> k29,
      MemRef<float, 4> k30, MemRef<float, 4> k31, MemRef<float, 4> k32,
      MemRef<float, 4> k33, MemRef<float, 4> k34, MemRef<float, 4> k35,
      MemRef<float, 4> k36, MemRef<float, 4> k37, MemRef<float, 4> k38,
      MemRef<float, 4> k39, MemRef<float, 4> k40, MemRef<float, 4> k41,
      MemRef<float, 4> k42, MemRef<float, 4> k43, MemRef<float, 4> k44,
      MemRef<float, 4> k45, MemRef<float, 4> k46, MemRef<float, 4> k47,
      MemRef<float, 4> k48, MemRef<float, 4> k49, MemRef<float, 4> k50,
      MemRef<float, 4> k51, MemRef<float, 4> k52, MemRef<float, 4> k53,
      MemRef<float, 4> k54, MemRef<float, 4> k55, MemRef<float, 3> l)
      : kv0(k0), kv1(k1), kv2(k2), kv3(k3), kv4(k4), kv5(k5), kv6(k6), kv7(k7),
        kv8(k8), kv9(k9), kv10(k10), kv11(k11), kv12(k12), kv13(k13), kv14(k14),
        kv15(k15), kv16(k16), kv17(k17), kv18(k18), kv19(k19), kv20(k20),
        kv21(k21), kv22(k22), kv23(k23), kv24(k24), kv25(k25), kv26(k26),
        kv27(k27), kv28(k28), kv29(k29), kv30(k30), kv31(k31), kv32(k32),
        kv33(k33), kv34(k34), kv35(k35), kv36(k36), kv37(k37), kv38(k38),
        kv39(k39), kv40(k40), kv41(k41), kv42(k42), kv43(k43), kv44(k44),
        kv45(k45), kv46(k46), kv47(k47), kv48(k48), kv49(k49), kv50(k50),
        kv51(k51), kv52(k52), kv53(k53), kv54(k54), kv55(k55), logits(l),
        kv_ptrs{&kv0,  &kv1,  &kv2,  &kv3,  &kv4,  &kv5,  &kv6,  &kv7,

                &kv8,  &kv9,  &kv10, &kv11, &kv12, &kv13, &kv14, &kv15,

                &kv16, &kv17, &kv18, &kv19, &kv20, &kv21, &kv22, &kv23,

                &kv24, &kv25, &kv26, &kv27, &kv28, &kv29, &kv30, &kv31,

                &kv32, &kv33, &kv34, &kv35, &kv36, &kv37, &kv38, &kv39,

                &kv40, &kv41, &kv42, &kv43, &kv44, &kv45, &kv46, &kv47,

                &kv48, &kv49, &kv50, &kv51, &kv52, &kv53, &kv54, &kv55} {}
};

// Decode returns 85 values: [cache_pos, kv_a, kv_b] × 28 layers + logits.
// Laid out as raw bytes to match the MLIR sret struct exactly without needing
// MemRef default constructors (which are protected).
struct DecodeResultContainer {
  // MemRef<long long, 1>: 2 ptrs + offset + 1 size + 1 stride = 5 × i64 = 40
  // bytes MemRef<float, 4>:     2 ptrs + offset + 4 sizes + 4 strides = 11 ×
  // i64 = 88 bytes MemRef<float, 3>:     2 ptrs + offset + 3 sizes + 3 strides
  // = 9 × i64 = 72 bytes Per layer: 40 + 88 + 88 = 216 bytes. 28 layers = 6048.
  // + 72 logits = 6120.
  static constexpr size_t kMemRef1xi64Size = 40;
  static constexpr size_t kMemRef4xf32Size = 88;
  static constexpr size_t kMemRef3xf32Size = 72;
  static constexpr size_t kLayerSize = kMemRef1xi64Size + 2 * kMemRef4xf32Size;
  static constexpr size_t kNumLayerGroups = NUM_LAYERS / 2; // 28
  static constexpr size_t kTotalSize =
      kNumLayerGroups * kLayerSize + kMemRef3xf32Size;

  alignas(16) char data[kTotalSize];

  MemRef<float, 4> &kv(int i) {
    int group = i / 2;
    int which = i % 2;
    size_t off =
        group * kLayerSize + kMemRef1xi64Size + which * kMemRef4xf32Size;
    return *reinterpret_cast<MemRef<float, 4> *>(data + off);
  }

  MemRef<float, 3> &logits() {
    size_t off = kNumLayerGroups * kLayerSize;
    return *reinterpret_cast<MemRef<float, 3> *>(data + off);
  }
};

extern "C" void _mlir_ciface_forward_prefill(MemRefContainer *result,
                                             MemRef<float, 1> *arg0,
                                             MemRef<long long, 2> *arg1);

// forward_decode takes 87 args: params, tokens, cache_pos, then [kv_a, kv_b,
// cache_idx] × 28.
extern "C" void _mlir_ciface_forward_decode(
    DecodeResultContainer *result, MemRef<float, 1> *params,
    MemRef<long long, 2> *tokens, MemRef<long long, 1> *cachePos,
    // 28 groups of: kv_a, kv_b, cache_idx
    MemRef<float, 4> *kv0, MemRef<float, 4> *kv1, MemRef<long long, 1> *idx0,
    MemRef<float, 4> *kv2, MemRef<float, 4> *kv3, MemRef<long long, 1> *idx1,
    MemRef<float, 4> *kv4, MemRef<float, 4> *kv5, MemRef<long long, 1> *idx2,
    MemRef<float, 4> *kv6, MemRef<float, 4> *kv7, MemRef<long long, 1> *idx3,
    MemRef<float, 4> *kv8, MemRef<float, 4> *kv9, MemRef<long long, 1> *idx4,
    MemRef<float, 4> *kv10, MemRef<float, 4> *kv11, MemRef<long long, 1> *idx5,
    MemRef<float, 4> *kv12, MemRef<float, 4> *kv13, MemRef<long long, 1> *idx6,
    MemRef<float, 4> *kv14, MemRef<float, 4> *kv15, MemRef<long long, 1> *idx7,
    MemRef<float, 4> *kv16, MemRef<float, 4> *kv17, MemRef<long long, 1> *idx8,
    MemRef<float, 4> *kv18, MemRef<float, 4> *kv19, MemRef<long long, 1> *idx9,
    MemRef<float, 4> *kv20, MemRef<float, 4> *kv21, MemRef<long long, 1> *idx10,
    MemRef<float, 4> *kv22, MemRef<float, 4> *kv23, MemRef<long long, 1> *idx11,
    MemRef<float, 4> *kv24, MemRef<float, 4> *kv25, MemRef<long long, 1> *idx12,
    MemRef<float, 4> *kv26, MemRef<float, 4> *kv27, MemRef<long long, 1> *idx13,
    MemRef<float, 4> *kv28, MemRef<float, 4> *kv29, MemRef<long long, 1> *idx14,
    MemRef<float, 4> *kv30, MemRef<float, 4> *kv31, MemRef<long long, 1> *idx15,
    MemRef<float, 4> *kv32, MemRef<float, 4> *kv33, MemRef<long long, 1> *idx16,
    MemRef<float, 4> *kv34, MemRef<float, 4> *kv35, MemRef<long long, 1> *idx17,
    MemRef<float, 4> *kv36, MemRef<float, 4> *kv37, MemRef<long long, 1> *idx18,
    MemRef<float, 4> *kv38, MemRef<float, 4> *kv39, MemRef<long long, 1> *idx19,
    MemRef<float, 4> *kv40, MemRef<float, 4> *kv41, MemRef<long long, 1> *idx20,
    MemRef<float, 4> *kv42, MemRef<float, 4> *kv43, MemRef<long long, 1> *idx21,
    MemRef<float, 4> *kv44, MemRef<float, 4> *kv45, MemRef<long long, 1> *idx22,
    MemRef<float, 4> *kv46, MemRef<float, 4> *kv47, MemRef<long long, 1> *idx23,
    MemRef<float, 4> *kv48, MemRef<float, 4> *kv49, MemRef<long long, 1> *idx24,
    MemRef<float, 4> *kv50, MemRef<float, 4> *kv51, MemRef<long long, 1> *idx25,
    MemRef<float, 4> *kv52, MemRef<float, 4> *kv53, MemRef<long long, 1> *idx26,
    MemRef<float, 4> *kv54, MemRef<float, 4> *kv55,
    MemRef<long long, 1> *idx27);

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

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
  total_time += time;
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

/// Tokenize input data in the container.
void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer) {
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
            << std::endl;
  const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

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
}

/// Find the index of the max value.
int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

void copy_kv_by_cache_position_block(const MemRefContainer &prefill,
                                     MemRefContainer &decode,
                                     int cache_position) {
  constexpr int num_kv = 56;
  int copy_len = std::min(cache_position, (int)MaxTokenLength);

  for (int k = 0; k < num_kv; ++k) {
    auto &src = *prefill.kv_ptrs[k];
    auto &dst = *decode.kv_ptrs[k];

    for (int h = 0; h < (int)HeadNum; ++h) {
      size_t bytes_to_copy =
          static_cast<size_t>(copy_len) * HiddenSize * sizeof(float);

      float *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
      float *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;

      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    }
  }
}

// -----------------------------------------------------------------------------
// DeepSeekR1 GPU Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title =
      "DeepSeekR1 GPU Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vocabulary and parameter files.
  /// DEEPSEEKR1_GPU_EXAMPLE_PATH points to the CPU source directory
  /// (vocab.txt). DEEPSEEKR1_GPU_EXAMPLE_BUILD_PATH points to the GPU build
  /// directory (arg0.data).
  std::string deepSeekR1Dir = DEEPSEEKR1_GPU_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_GPU_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string paramsDir = deepSeekR1BuildDir + "arg0.data";

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  CudaManagedMemRef<long long, 2> prefillTokensGPU({1, MaxTokenLength}, 0LL);
  CudaManagedMemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  CudaManagedMemRef1D ParamsContainer(ParamsSize);
  CudaManagedMemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<float, 3> logits_prefill({1, MaxTokenLength, MaxVocabSize});

  CudaManagedMemRef<float, 4> kv0({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);
  CudaManagedMemRef<float, 4> kv1({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);
  CudaManagedMemRef<float, 4> kv2({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);
  CudaManagedMemRef<float, 4> kv3({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);
  CudaManagedMemRef<float, 4> kv4({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);
  CudaManagedMemRef<float, 4> kv5({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);
  CudaManagedMemRef<float, 4> kv6({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);
  CudaManagedMemRef<float, 4> kv7({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);

  CudaManagedMemRef<float, 4> kv8({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);
  CudaManagedMemRef<float, 4> kv9({1, HeadNum, MaxTokenLength, HiddenSize},
                                  0.0f);
  CudaManagedMemRef<float, 4> kv10({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv11({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv12({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv13({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv14({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv15({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);

  CudaManagedMemRef<float, 4> kv16({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv17({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv18({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv19({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv20({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv21({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv22({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv23({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);

  CudaManagedMemRef<float, 4> kv24({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv25({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv26({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv27({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv28({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv29({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv30({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv31({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);

  CudaManagedMemRef<float, 4> kv32({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv33({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv34({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv35({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv36({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv37({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv38({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv39({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);

  CudaManagedMemRef<float, 4> kv40({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv41({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv42({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv43({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv44({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv45({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv46({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv47({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);

  CudaManagedMemRef<float, 4> kv48({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv49({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv50({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv51({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv52({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv53({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv54({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);
  CudaManagedMemRef<float, 4> kv55({1, HeadNum, MaxTokenLength, HiddenSize},
                                   0.0f);

  MemRefContainer prefillResultContainer(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logits_prefill);
  MemRefContainer *ptrPrefillResultContainer = &prefillResultContainer;

  /// Fill data into containers
  tokenizeInput(vocabDir, inputContainerPrefill);
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, ParamsContainer);

  // Copy prefill token IDs into GPU-accessible managed memory.
  // Text<size_t,2> uses regular malloc (CPU-only); GPU kernels cannot access
  // it directly. prefillTokensGPU uses cudaMallocManaged so the same virtual
  // address is valid on both CPU and GPU.
  {
    size_t tokenCnt = static_cast<size_t>(inputContainerPrefill.getTokenCnt());
    for (size_t i = 0; i < tokenCnt; ++i)
      prefillTokensGPU.getData()[i] =
          static_cast<long long>(inputContainerPrefill.getData()[i]);
  }

  /// Run DeepSeekR1 GPU Inference
  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(ptrPrefillResultContainer, &ParamsContainer,
                               &prefillTokensGPU);
  cudaDeviceSynchronize();
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
  const float *startPtr =
      ptrPrefillResultContainer->logits.getData() + tokenIndex * MaxVocabSize;
  const float *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);
  std::string tok = inputContainerPrefill.getStr(maxIndex);
  printIterInfo(0, tok, inferenceTime.count() / 1000);
  const double prefillSeconds = inferenceTime.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
  }
  inputContainerDecode.getData()[0] = (long long)maxIndex;
  outputContainer.appendTokenIdx(maxIndex);

  DecodeResultContainer decodeResult;
  DecodeResultContainer *ptrDecodeResult = &decodeResult;

  // Allocate decode KV caches via the MLIR CUDA runtime (same allocator as
  // GPU subgraphs) and copy prefill results into them.
  {
    const size_t kvElems = 1 * HeadNum * MaxTokenLength * HiddenSize;
    const size_t kvBytes = kvElems * sizeof(float);
    const intptr_t kvSizes[4] = {1, (intptr_t)HeadNum, (intptr_t)MaxTokenLength,
                                 (intptr_t)HiddenSize};
    const intptr_t kvStrides[4] = {
        (intptr_t)(HeadNum * MaxTokenLength * HiddenSize),
        (intptr_t)(MaxTokenLength * HiddenSize), (intptr_t)HiddenSize, 1};
    int copy_len =
        std::min((int)inputContainerPrefill.getTokenCnt(), (int)MaxTokenLength);
    for (int k = 0; k < (int)NUM_LAYERS; ++k) {
      float *buf = (float *)mgpuMemAlloc(kvBytes, /*stream=*/nullptr,
                                         /*isHostShared=*/false);
      std::memset(buf, 0, kvBytes);
      // Write the MemRef descriptor directly (fields are protected).
      // Layout: allocated(8), aligned(8), offset(8), sizes[4](32),
      // strides[4](32)
      char *desc = reinterpret_cast<char *>(&ptrDecodeResult->kv(k));
      intptr_t zero = 0;
      std::memcpy(desc + 0, &buf, 8);        // allocated
      std::memcpy(desc + 8, &buf, 8);        // aligned
      std::memcpy(desc + 16, &zero, 8);      // offset
      std::memcpy(desc + 24, kvSizes, 32);   // sizes[4]
      std::memcpy(desc + 56, kvStrides, 32); // strides[4]
      // Copy prefill KV data.
      auto &src = *prefillResultContainer.kv_ptrs[k];
      cudaDeviceSynchronize();
      for (int h = 0; h < (int)HeadNum; ++h) {
        size_t bytes =
            static_cast<size_t>(copy_len) * HiddenSize * sizeof(float);
        float *sp = src.getData() + h * MaxTokenLength * HiddenSize;
        float *dp = buf + h * MaxTokenLength * HiddenSize;
        std::memcpy(dp, sp, bytes);
      }
    }
  }

  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
  int generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;
  for (int i = 1; i <= generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_decode(
        ptrDecodeResult, &ParamsContainer, &inputContainerDecode,
        &cachePosition, &ptrDecodeResult->kv(0), &ptrDecodeResult->kv(1),
        &cachePosition, &ptrDecodeResult->kv(2), &ptrDecodeResult->kv(3),
        &cachePosition, &ptrDecodeResult->kv(4), &ptrDecodeResult->kv(5),
        &cachePosition, &ptrDecodeResult->kv(6), &ptrDecodeResult->kv(7),
        &cachePosition, &ptrDecodeResult->kv(8), &ptrDecodeResult->kv(9),
        &cachePosition, &ptrDecodeResult->kv(10), &ptrDecodeResult->kv(11),
        &cachePosition, &ptrDecodeResult->kv(12), &ptrDecodeResult->kv(13),
        &cachePosition, &ptrDecodeResult->kv(14), &ptrDecodeResult->kv(15),
        &cachePosition, &ptrDecodeResult->kv(16), &ptrDecodeResult->kv(17),
        &cachePosition, &ptrDecodeResult->kv(18), &ptrDecodeResult->kv(19),
        &cachePosition, &ptrDecodeResult->kv(20), &ptrDecodeResult->kv(21),
        &cachePosition, &ptrDecodeResult->kv(22), &ptrDecodeResult->kv(23),
        &cachePosition, &ptrDecodeResult->kv(24), &ptrDecodeResult->kv(25),
        &cachePosition, &ptrDecodeResult->kv(26), &ptrDecodeResult->kv(27),
        &cachePosition, &ptrDecodeResult->kv(28), &ptrDecodeResult->kv(29),
        &cachePosition, &ptrDecodeResult->kv(30), &ptrDecodeResult->kv(31),
        &cachePosition, &ptrDecodeResult->kv(32), &ptrDecodeResult->kv(33),
        &cachePosition, &ptrDecodeResult->kv(34), &ptrDecodeResult->kv(35),
        &cachePosition, &ptrDecodeResult->kv(36), &ptrDecodeResult->kv(37),
        &cachePosition, &ptrDecodeResult->kv(38), &ptrDecodeResult->kv(39),
        &cachePosition, &ptrDecodeResult->kv(40), &ptrDecodeResult->kv(41),
        &cachePosition, &ptrDecodeResult->kv(42), &ptrDecodeResult->kv(43),
        &cachePosition, &ptrDecodeResult->kv(44), &ptrDecodeResult->kv(45),
        &cachePosition, &ptrDecodeResult->kv(46), &ptrDecodeResult->kv(47),
        &cachePosition, &ptrDecodeResult->kv(48), &ptrDecodeResult->kv(49),
        &cachePosition, &ptrDecodeResult->kv(50), &ptrDecodeResult->kv(51),
        &cachePosition, &ptrDecodeResult->kv(52), &ptrDecodeResult->kv(53),
        &cachePosition, &ptrDecodeResult->kv(54), &ptrDecodeResult->kv(55),
        &cachePosition);
    cudaDeviceSynchronize();

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    decodeTimeAccumMs += inferenceTime.count();
    decodeTokens += 1;

    // Determine the generated token.
    const float *startPtr = ptrDecodeResult->logits().getData();
    const float *endPtr = startPtr + MaxVocabSize;
    maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainerPrefill.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // Stop if a <|end▁of▁sentence|> token is generated.
    if (maxIndex == 151643) {
      break;
    }
    // Append the generated token into the input and output container.
    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    cachePosition.getData()[0] += 1;
  }

  const double decodeSeconds = decodeTimeAccumMs / 1000.0;
  const double decodeTokensPerSec =
      decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
                          : 0.0;

  /// Print the final result
  std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
  std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m "
            << outputContainer.revertDeepSeekR1() << std::endl;

  std::cout.flush();
  // Use _Exit to skip MLIR CUDA runtime atexit handlers which attempt to call
  // cuModuleUnload after the CUDA context has already been destroyed.
  std::_Exit(0);
}
