//===buddy-deepseek-r1-distributed.cpp-------------------------------------===//
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
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <mpi.h>
#include <string>
#include <vector>

using namespace buddy;
double total_time = 0;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;
constexpr size_t SubMaxTokenLength = 512;

constexpr size_t VocabShardSize = MaxVocabSize / 2;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HiddenSize0 = 1536;
constexpr size_t HeadNum = 2;
constexpr int FrontendRank = 0;
constexpr int PeerRank = 1;

struct MemRefContainer0 {
  MemRef<float, 3> data;
  MemRef<int8_t, 4> mask;
  MemRef<float, 3> cos;
  MemRef<float, 3> sin;

  MemRefContainer0(MemRef<float, 3> m1, MemRef<int8_t, 4> m2,
                   MemRef<float, 3> m3, MemRef<float, 3> m4)
      : data(m1), mask(m2), cos(m3), sin(m4) {}
};

struct MemRefContainer2 {
  MemRef<float, 4> kcache;
  MemRef<float, 4> vcache;
  MemRef<float, 2> data;

  MemRefContainer2(MemRef<float, 4> m1, MemRef<float, 4> m2,
                   MemRef<float, 2> m3)
      : kcache(m1), vcache(m2), data(m3) {}
};

struct MemRefContainerDecode2 {
  MemRef<long long, 1> cache_position_out;
  MemRef<float, 4> kcache;
  MemRef<float, 4> vcache;
  MemRef<float, 2> data;

  MemRefContainerDecode2(MemRef<long long, 1> m0, MemRef<float, 4> m1,
                         MemRef<float, 4> m2, MemRef<float, 2> m3)
      : cache_position_out(m0), kcache(m1), vcache(m2), data(m3) {}
};

/// Declare DeepSeekR1 forward function.
extern "C" {
void _mlir_ciface_forward_prefill0(MemRefContainer0 *, MemRef<float, 1> *,
                                   Text<size_t, 2> *);
void _mlir_ciface_forward_prefill1(MemRef<float, 3> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill2(MemRefContainer2 *, MemRef<float, 1> *,
                                   MemRef<int8_t, 4> *, MemRef<float, 3> *,
                                   MemRef<float, 3> *, MemRef<float, 3> *);
void _mlir_ciface_forward_prefill3(MemRef<float, 3> *, MemRef<float, 3> *,
                                   MemRef<float, 2> *);
void _mlir_ciface_forward_prefill5(MemRef<float, 2> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill169(MemRef<float, 3> *, MemRef<float, 1> *,
                                     MemRef<float, 3> *);

void _mlir_ciface_forward_decode0(MemRefContainer0 *, MemRef<float, 1> *,
                                  MemRef<long long, 2> *,
                                  MemRef<long long, 1> *);
void _mlir_ciface_forward_decode1(MemRef<float, 3> *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode2(MemRefContainerDecode2 *, MemRef<float, 1> *,
                                  MemRef<long long, 1> *, MemRef<float, 4> *,
                                  MemRef<float, 4> *, MemRef<int8_t, 4> *,
                                  MemRef<float, 3> *, MemRef<float, 3> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode3(MemRef<float, 3> *, MemRef<float, 3> *,
                                  MemRef<float, 2> *);
void _mlir_ciface_forward_decode5(MemRef<float, 2> *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode169(MemRef<float, 3> *, MemRef<float, 1> *,
                                    MemRef<float, 3> *);
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

using HighResClock = std::chrono::high_resolution_clock;

struct MaxLocPair {
  float value = -std::numeric_limits<float>::infinity();
  int index = -1;
};

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
void printIterInfo(size_t iterIdx, const std::string &str, double time) {
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
  const auto buddyTokenizeStart = HighResClock::now();
  inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = HighResClock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = HighResClock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    std::cout << paramFilePath << std::endl;
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
  const auto loadEnd = HighResClock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << loadTime.count() / 1000.0 << "s\n"
            << std::endl;
}

/// Find the index of the max value.
int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

inline void addFloatVectorInPlace(float *dst, const float *src, int count) {
  for (int i = 0; i < count; ++i) {
    dst[i] += src[i];
  }
}

inline void reduceSumTwoRanksSendrecv(const float *localBuf, float *sumBuf,
                                      int count, int peerRankInComm, int tag,
                                      MPI_Comm comm) {
  if (comm == MPI_COMM_NULL) {
    std::memcpy(sumBuf, localBuf, sizeof(float) * count);
    return;
  }
  MPI_Sendrecv(localBuf, count, MPI_FLOAT, peerRankInComm, tag, sumBuf, count,
               MPI_FLOAT, peerRankInComm, tag, comm, MPI_STATUS_IGNORE);
  addFloatVectorInPlace(sumBuf, localBuf, count);
}

inline MaxLocPair reduceMaxLoc(float localValue, int globalIndex,
                               MPI_Comm comm) {
  if (comm == MPI_COMM_NULL) {
    return MaxLocPair{localValue, globalIndex};
  }

  struct {
    float value;
    int index;
  } local{localValue, globalIndex}, global{0.0f, -1};

  MPI_Allreduce(&local, &global, 1, MPI_FLOAT_INT, MPI_MAXLOC, comm);
  return MaxLocPair{global.value, global.index};
}

// -----------------------------------------------------------------------------
// DeepSeekR1 Inference Main Entry
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {

  /// Define directories of vocabulary and parameter file.
  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "/vocab.txt";

  int subSize = SubMaxTokenLength * HiddenSize0;
  int offset0 = subSize;

  int rank, size;
  int generateLen = MaxTokenLength;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (rank == FrontendRank) {
      std::cerr << "[Error] This intermediate version requires exactly 2 MPI "
                   "ranks.\n";
    }
    MPI_Finalize();
    return 1;
  }

  if (rank == FrontendRank) {
    const std::string title = "DeepSeekR1  Inference Powered by Buddy Compiler";
    std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

    // -----------------------------------------------------------------------
    // Rank 0 frontend resources.
    // -----------------------------------------------------------------------
    Text<size_t, 2> outputContainer;
    outputContainer.loadVocab(vocabDir);

    MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
    MemRef<long long, 1> cachePosition({1}, 0LL);

    MemRef<float, 3> myMemRef_decode1({1, 1, HiddenSize0});
    MemRef<int8_t, 4> myMemRef_decode2({1, 1, 1, MaxTokenLength});
    MemRef<float, 3> myMemRef_decode3({1, 1, HiddenSize});
    MemRef<float, 3> myMemRef_decode4({1, 1, HiddenSize});
    MemRefContainer0 resultContainerDecode(myMemRef_decode1, myMemRef_decode2,
                                           myMemRef_decode3, myMemRef_decode4);
    MemRefContainer0 *resultContainerDecodePtr = &resultContainerDecode;
    MemRef<float, 3> resultDecodeShard0({1, 1, VocabShardSize});

    // -----------------------------------------------------------------------
    // Rank 0 local TP shard resources.
    // -----------------------------------------------------------------------
    std::vector<MemRef<float, 4>> kv0;
    kv0.reserve(NUM_LAYERS);
    for (size_t i = 0; i < NUM_LAYERS; ++i) {
      kv0.emplace_back(std::vector<size_t>{1, 1, MaxTokenLength, HiddenSize});
    }

    MemRef<float, 3> subResultContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 3> sub3DContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 2> tmp2DContainerDecode({1, HiddenSize0});
    MemRef<float, 2> sub2DContainerDecode({1, HiddenSize0});
    float *mhaOutputPtrDecode = tmp2DContainerDecode.getData();
    MemRef<long long, 1> cachePositionOutDecode({1}, 0LL);
    MemRefContainerDecode2 kvDecodeContainer0(cachePositionOutDecode, kv0[0],
                                              kv0[1], tmp2DContainerDecode);
    MemRefContainerDecode2 *kvDecodeContainerPtr0 = &kvDecodeContainer0;

    // -----------------------------------------------------------------------
    // Load frontend and local-shard params.
    // -----------------------------------------------------------------------
    constexpr size_t param_size0 = 233373760;
    const std::string paramsDir0 =
        deepSeekR1BuildDir + "/subgraph0_prefill0_arg0.data";
    constexpr size_t param_size2 = 116688384;
    const std::string paramsDir2 =
        deepSeekR1BuildDir + "/subgraph0_decode169_arg0.data";
    MemRef<float, 1> paramsContainer0({param_size0});
    loadParameters(paramsDir0, paramsContainer0);
    MemRef<float, 1> paramsContainer2({param_size2});
    loadParameters(paramsDir2, paramsContainer2);

    constexpr size_t paramSizeRMS = 1536;
    constexpr size_t paramSizeMHA = 2753536;
    constexpr size_t paramSizeMLP = 20643840;
    int times = 28;

    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA, paramsContainersMLP;

    for (int i = 1; i < 169; i += 6) {
      paramsDirsRMS.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0.data");
      paramsDirsRMS0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 3) + "_arg0.data");
    }
    for (int i = 2; i < 169; i += 6) {
      paramsDirsMHA.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0.data");
      paramsDirsMLP.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i + 3) + "_arg0.data");
    }

    for (int i = 0; i < times; i++) {
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);

      MemRef<float, 1> paramsContainerMHA({paramSizeMHA});
      loadParameters(paramsDirsMHA[i], paramsContainerMHA);
      paramsContainersMHA.push_back(paramsContainerMHA);

      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0);
      paramsContainersRMS0.push_back(paramsContainerRMS0);

      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    MPI_Comm comm_sub = MPI_COMM_NULL;
    MPI_Group world_group = MPI_GROUP_NULL;
    MPI_Group sub_group = MPI_GROUP_NULL;
    int ranks_in_sub[2] = {FrontendRank, PeerRank};
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, 2, ranks_in_sub, &sub_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &comm_sub);

    // -----------------------------------------------------------------------
    // Input.
    // -----------------------------------------------------------------------
    std::string inputStr;
    getUserInput(inputStr);
    Text<size_t, 2> inputContainerPrefill(inputStr);
    tokenizeInput(vocabDir, inputContainerPrefill);
    std::cout << "Input token count: " << inputContainerPrefill.getTokenCnt()
              << std::endl;

    int maxIndex = 0;
    std::string tok;
    double prefillTokensPerSec = 0.0;
    double prefillSeconds = 0.0;
    float *inputPtr = nullptr;

    {
      MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize0});
      MemRef<int8_t, 4> myMemRef2({1, 1, MaxTokenLength, MaxTokenLength});
      MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize});
      MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize});
      MemRefContainer0 resultContainer(myMemRef1, myMemRef2, myMemRef3,
                                       myMemRef4);
      MemRefContainer0 *resultContainerPtr = &resultContainer;

      MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize0});
      MemRef<float, 3> resultPrefill({1, MaxTokenLength, MaxVocabSize});

      MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize0});
      MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize0});
      MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize0});
      MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize0});
      MemRefContainer2 kvContainer0(kv0[0], kv0[1], tmp2DContainer);
      MemRefContainer2 *kvContainerPtr0 = &kvContainer0;

      constexpr size_t param_size1 = 233375232;
      const std::string paramsDir1 =
          deepSeekR1BuildDir + "/subgraph0_prefill169_arg0.data";
      MemRef<float, 1> paramsContainer1({param_size1});
      loadParameters(paramsDir1, paramsContainer1);

      MPI_Request send_req_hidden = MPI_REQUEST_NULL;
      MPI_Request send_req_mha[3];
      MPI_Request recv_req_prefill = MPI_REQUEST_NULL;

      float *outputPtr = tmp3DMemRef.getData();
      float *subResultPtr = subResultContainer.getData();
      float *rmsPtr = sub3DContainer.getData();
      float *mhaOutputPtr = tmp2DContainer.getData();
      float *sub2DPtr = sub2DContainer.getData();

      const auto inferenceStart = HighResClock::now();

      std::cout << "\n\033[33;1m[Inference Start]\033[0m" << std::endl;

      _mlir_ciface_forward_prefill0(resultContainerPtr, &paramsContainer0,
                                    &inputContainerPrefill);
      inputPtr = resultContainerPtr->data.getData();

      std::memcpy(subResultPtr, inputPtr, sizeof(float) * subSize);

      MPI_Isend(inputPtr + offset0, subSize, MPI_FLOAT, PeerRank, 0,
                MPI_COMM_WORLD, &send_req_hidden);
      MPI_Isend(resultContainerPtr->mask.getData(),
                MaxTokenLength * MaxTokenLength, MPI_INT8_T, PeerRank, 1,
                MPI_COMM_WORLD, &send_req_mha[0]);
      MPI_Isend(resultContainerPtr->cos.getData(), MaxTokenLength * HiddenSize,
                MPI_FLOAT, PeerRank, 2, MPI_COMM_WORLD, &send_req_mha[1]);
      MPI_Isend(resultContainerPtr->sin.getData(), MaxTokenLength * HiddenSize,
                MPI_FLOAT, PeerRank, 3, MPI_COMM_WORLD, &send_req_mha[2]);
      MPI_Irecv(outputPtr + offset0, subSize, MPI_FLOAT, PeerRank, 0,
                MPI_COMM_WORLD, &recv_req_prefill);

      for (int m = 0; m < times; m++) {
        _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS[m],
                                      &subResultContainer);
        rmsPtr = sub3DContainer.getData();

        if (comm_sub != MPI_COMM_NULL) {
          MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                        subSize, MPI_FLOAT, comm_sub);
        }

        _mlir_ciface_forward_prefill2(
            kvContainerPtr0, &paramsContainersMHA[m], &resultContainerPtr->mask,
            &resultContainerPtr->cos, &resultContainerPtr->sin, &tmp3DMemRef);
        kv0[2 * m] = kvContainerPtr0->kcache;
        kv0[2 * m + 1] = kvContainerPtr0->vcache;
        tmp2DContainer = kvContainerPtr0->data;
        mhaOutputPtr = tmp2DContainer.getData();

        if (comm_sub != MPI_COMM_NULL) {
          MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                   MPI_SUM, comm_sub);
        }

        _mlir_ciface_forward_prefill3(&subResultContainer, &subResultContainer,
                                      &sub2DContainer);

        _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS0[m],
                                      &subResultContainer);
        rmsPtr = sub3DContainer.getData();

        if (comm_sub != MPI_COMM_NULL) {
          MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                        subSize, MPI_FLOAT, comm_sub);
        }

        _mlir_ciface_forward_prefill5(&tmp2DContainer, &paramsContainersMLP[m],
                                      &tmp3DMemRef);
        mhaOutputPtr = tmp2DContainer.getData();

        if (comm_sub != MPI_COMM_NULL) {
          MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                   MPI_SUM, comm_sub);
        }

        _mlir_ciface_forward_prefill3(&subResultContainer, &subResultContainer,
                                      &sub2DContainer);
      }

      std::memcpy(outputPtr, subResultContainer.getData(),
                  sizeof(float) * subSize);
      MPI_Wait(&recv_req_prefill, MPI_STATUS_IGNORE);
      MPI_Wait(&send_req_hidden, MPI_STATUS_IGNORE);
      MPI_Waitall(3, send_req_mha, MPI_STATUSES_IGNORE);

      _mlir_ciface_forward_prefill169(&resultPrefill, &paramsContainer1,
                                      &tmp3DMemRef);

      const auto inferenceEnd = HighResClock::now();
      const std::chrono::duration<double> inferenceTime =
          inferenceEnd - inferenceStart;
      prefillSeconds = inferenceTime.count();

      int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
      const float *startPtr =
          resultPrefill.getData() + tokenIndex * MaxVocabSize;
      const float *endPtr = startPtr + MaxVocabSize;
      maxIndex = findMaxIndex(startPtr, endPtr);
      tok = inputContainerPrefill.getStr(maxIndex);
      printIterInfo(0, tok, prefillSeconds);

      if (prefillSeconds > 0.0) {
        prefillTokensPerSec =
            static_cast<double>(MaxTokenLength) / prefillSeconds;
      }

      inputContainerDecode.getData()[0] = static_cast<long long>(maxIndex);
      outputContainer.appendTokenIdx(maxIndex);
      cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
      generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
    }

    double decodeTimeAccumMs = 0.0;
    size_t decodeTokens = 0;

    MPI_Request send_req_decode[5];
    int ctrl = 0;
    bool sentStop = false;

    for (int i = 1; i <= generateLen; i++) {
      ctrl = 0;
      MPI_Send(&ctrl, 1, MPI_INT, PeerRank, 99, MPI_COMM_WORLD);

      const auto decodeStart = HighResClock::now();

      _mlir_ciface_forward_decode0(resultContainerDecodePtr, &paramsContainer0,
                                   &inputContainerDecode, &cachePosition);
      inputPtr = resultContainerDecodePtr->data.getData();

      MPI_Isend(inputPtr, HiddenSize0, MPI_FLOAT, PeerRank, 0, MPI_COMM_WORLD,
                &send_req_decode[0]);
      MPI_Isend(resultContainerDecodePtr->mask.getData(), MaxTokenLength,
                MPI_INT8_T, PeerRank, 1, MPI_COMM_WORLD, &send_req_decode[1]);
      MPI_Isend(resultContainerDecodePtr->cos.getData(), HiddenSize, MPI_FLOAT,
                PeerRank, 2, MPI_COMM_WORLD, &send_req_decode[2]);
      MPI_Isend(resultContainerDecodePtr->sin.getData(), HiddenSize, MPI_FLOAT,
                PeerRank, 3, MPI_COMM_WORLD, &send_req_decode[3]);
      MPI_Isend(cachePosition.getData(), 1, MPI_LONG_LONG, PeerRank, 4,
                MPI_COMM_WORLD, &send_req_decode[4]);

      std::memcpy(subResultContainerDecode.getData(), inputPtr,
                  sizeof(float) * HiddenSize0);

      for (int m = 0; m < times; m++) {
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS[m],
                                     &subResultContainerDecode);

        _mlir_ciface_forward_decode2(
            kvDecodeContainerPtr0, &paramsContainersMHA[m], &cachePosition,
            &kv0[2 * m], &kv0[2 * m + 1], &resultContainerDecodePtr->mask,
            &resultContainerDecodePtr->cos, &resultContainerDecodePtr->sin,
            &sub3DContainerDecode);
        kv0[2 * m] = kvDecodeContainerPtr0->kcache;
        kv0[2 * m + 1] = kvDecodeContainerPtr0->vcache;
        tmp2DContainerDecode = kvDecodeContainerPtr0->data;
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        if (comm_sub != MPI_COMM_NULL) {
          reduceSumTwoRanksSendrecv(mhaOutputPtrDecode,
                                    sub2DContainerDecode.getData(), HiddenSize0,
                                    PeerRank, 1000 + m, comm_sub);
        } else {
          std::memcpy(sub2DContainerDecode.getData(), mhaOutputPtrDecode,
                      sizeof(float) * HiddenSize0);
        }

        _mlir_ciface_forward_decode3(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);

        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS0[m],
                                     &subResultContainerDecode);

        _mlir_ciface_forward_decode5(&tmp2DContainerDecode,
                                     &paramsContainersMLP[m],
                                     &sub3DContainerDecode);
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        if (comm_sub != MPI_COMM_NULL) {
          reduceSumTwoRanksSendrecv(mhaOutputPtrDecode,
                                    sub2DContainerDecode.getData(), HiddenSize0,
                                    PeerRank, 2000 + m, comm_sub);
        } else {
          std::memcpy(sub2DContainerDecode.getData(), mhaOutputPtrDecode,
                      sizeof(float) * HiddenSize0);
        }

        _mlir_ciface_forward_decode3(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
      }

      MPI_Waitall(5, send_req_decode, MPI_STATUSES_IGNORE);

      std::memcpy(myMemRef_decode1.getData(),
                  subResultContainerDecode.getData(),
                  sizeof(float) * HiddenSize0);

      _mlir_ciface_forward_decode169(&resultDecodeShard0, &paramsContainer2,
                                     &myMemRef_decode1);

      const float *decodeShard0StartPtr = resultDecodeShard0.getData();
      const float *decodeShard0EndPtr = decodeShard0StartPtr + VocabShardSize;
      int localDecodeMaxIndex =
          findMaxIndex(decodeShard0StartPtr, decodeShard0EndPtr);
      float localDecodeMaxValue = decodeShard0StartPtr[localDecodeMaxIndex];

      MaxLocPair globalTop1 =
          reduceMaxLoc(localDecodeMaxValue, localDecodeMaxIndex, comm_sub);
      maxIndex = globalTop1.index;

      const auto decodeEnd = HighResClock::now();
      const std::chrono::duration<double, std::milli> decodeTime =
          decodeEnd - decodeStart;
      decodeTimeAccumMs += decodeTime.count();
      decodeTokens += 1;

      tok = inputContainerPrefill.getStr(maxIndex);
      printIterInfo(i, tok, decodeTime.count() / 1000.0);

      if (maxIndex == 151643) {
        ctrl = 1;
        MPI_Send(&ctrl, 1, MPI_INT, PeerRank, 99, MPI_COMM_WORLD);
        sentStop = true;
        break;
      }

      inputContainerDecode.getData()[0] = maxIndex;
      outputContainer.appendTokenIdx(maxIndex);
      cachePosition.getData()[0] += 1;
    }

    if (!sentStop) {
      ctrl = 1;
      MPI_Send(&ctrl, 1, MPI_INT, PeerRank, 99, MPI_COMM_WORLD);
    }

    double decodeSeconds = decodeTimeAccumMs / 1000.0;
    const double decodeTokensPerSec =
        decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
                            : 0.0;

    std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
    std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
              << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTokensPerSec
              << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
    std::cout << "\033[33;1m[Output]\033[0m "
              << outputContainer.revertDeepSeekR1() << std::endl;

    if (comm_sub != MPI_COMM_NULL) {
      MPI_Comm_free(&comm_sub);
    }
    if (sub_group != MPI_GROUP_NULL) {
      MPI_Group_free(&sub_group);
    }
    if (world_group != MPI_GROUP_NULL) {
      MPI_Group_free(&world_group);
    }

  } else if (rank == PeerRank) {

    std::vector<MemRef<float, 4>> kv0;
    kv0.reserve(NUM_LAYERS);
    for (size_t i = 0; i < NUM_LAYERS; ++i) {
      kv0.emplace_back(std::vector<size_t>{1, 1, MaxTokenLength, HiddenSize});
    }

    constexpr size_t paramSizeRMS = 1536;
    constexpr size_t paramSizeMHA = 2753536;
    constexpr size_t paramSizeMLP = 20643840;
    int times = 28;
    int source = FrontendRank;

    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA, paramsContainersMLP;

    for (int i = 1; i < 169; i += 6) {
      paramsDirsRMS.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0.data");
      paramsDirsRMS0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 3) + "_arg0.data");
    }
    for (int i = 2; i < 169; i += 6) {
      paramsDirsMHA.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg1.data");
      paramsDirsMLP.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i + 3) + "_arg1.data");
    }

    for (int i = 0; i < times; i++) {
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);

      MemRef<float, 1> paramsContainerMHA({paramSizeMHA});
      loadParameters(paramsDirsMHA[i], paramsContainerMHA);
      paramsContainersMHA.push_back(paramsContainerMHA);

      MemRef<float, 1> paramsContainerRMS0({paramSizeRMS});
      loadParameters(paramsDirsRMS0[i], paramsContainerRMS0);
      paramsContainersRMS0.push_back(paramsContainerRMS0);

      MemRef<float, 1> paramsContainerMLP({paramSizeMLP});
      loadParameters(paramsDirsMLP[i], paramsContainerMLP);
      paramsContainersMLP.push_back(paramsContainerMLP);
    }

    constexpr size_t param_size2 = 116688384;
    const std::string paramsDir2 =
        deepSeekR1BuildDir + "/subgraph0_decode169_arg1.data";
    MemRef<float, 1> paramsContainer2({param_size2});
    loadParameters(paramsDir2, paramsContainer2);

    MPI_Comm comm_sub = MPI_COMM_NULL;
    MPI_Group world_group = MPI_GROUP_NULL;
    MPI_Group sub_group = MPI_GROUP_NULL;
    int ranks_in_sub[2] = {FrontendRank, PeerRank};
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, 2, ranks_in_sub, &sub_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &comm_sub);

    {
      MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize0});
      MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize0});
      MemRef<int8_t, 4> mhaMemRef4D({1, 1, MaxTokenLength, MaxTokenLength});
      MemRef<float, 3> mhaMemRef3D1({1, MaxTokenLength, HiddenSize});
      MemRef<float, 3> mhaMemRef3D2({1, MaxTokenLength, HiddenSize});
      MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize0});
      MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize0});
      MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize0});
      MemRefContainer2 kvContainer0(kv0[0], kv0[1], tmp2DContainer);
      MemRefContainer2 *kvContainerPtr0 = &kvContainer0;

      float *subResultPtr = subResultContainer.getData();
      float *rmsPtr = sub3DContainer.getData();
      int8_t *mhaMemRef4DPtr = mhaMemRef4D.getData();
      float *mhaMemRef3D1Ptr = mhaMemRef3D1.getData();
      float *mhaMemRef3D2Ptr = mhaMemRef3D2.getData();
      float *mhaOutputPtr = tmp2DContainer.getData();
      float *sub2DPtr = sub2DContainer.getData();

      MPI_Request recv_req_hidden = MPI_REQUEST_NULL;
      MPI_Request recv_req_mha[3];

      MPI_Irecv(mhaMemRef4DPtr, MaxTokenLength * MaxTokenLength, MPI_INT8_T,
                source, 1, MPI_COMM_WORLD, &recv_req_mha[0]);
      MPI_Irecv(mhaMemRef3D1Ptr, MaxTokenLength * HiddenSize, MPI_FLOAT, source,
                2, MPI_COMM_WORLD, &recv_req_mha[1]);
      MPI_Irecv(mhaMemRef3D2Ptr, MaxTokenLength * HiddenSize, MPI_FLOAT, source,
                3, MPI_COMM_WORLD, &recv_req_mha[2]);
      MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
                &recv_req_hidden);

      MPI_Wait(&recv_req_hidden, MPI_STATUS_IGNORE);
      MPI_Waitall(3, recv_req_mha, MPI_STATUSES_IGNORE);

      for (int m = 0; m < times; m++) {
        _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS[m],
                                      &subResultContainer);
        rmsPtr = sub3DContainer.getData();

        if (comm_sub != MPI_COMM_NULL) {
          MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                        subSize, MPI_FLOAT, comm_sub);
        }

        _mlir_ciface_forward_prefill2(kvContainerPtr0, &paramsContainersMHA[m],
                                      &mhaMemRef4D, &mhaMemRef3D1,
                                      &mhaMemRef3D2, &tmp3DMemRef);
        kv0[2 * m] = kvContainerPtr0->kcache;
        kv0[2 * m + 1] = kvContainerPtr0->vcache;
        tmp2DContainer = kvContainerPtr0->data;
        mhaOutputPtr = tmp2DContainer.getData();

        if (comm_sub != MPI_COMM_NULL) {
          MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                   MPI_SUM, comm_sub);
        }

        _mlir_ciface_forward_prefill3(&subResultContainer, &subResultContainer,
                                      &sub2DContainer);

        _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS0[m],
                                      &subResultContainer);
        rmsPtr = sub3DContainer.getData();

        if (comm_sub != MPI_COMM_NULL) {
          MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                        subSize, MPI_FLOAT, comm_sub);
        }

        _mlir_ciface_forward_prefill5(&tmp2DContainer, &paramsContainersMLP[m],
                                      &tmp3DMemRef);
        mhaOutputPtr = tmp2DContainer.getData();

        if (comm_sub != MPI_COMM_NULL) {
          MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                   MPI_SUM, comm_sub);
        }

        _mlir_ciface_forward_prefill3(&subResultContainer, &subResultContainer,
                                      &sub2DContainer);
      }

      subResultPtr = subResultContainer.getData();
      MPI_Send(subResultPtr, subSize, MPI_FLOAT, FrontendRank, 0,
               MPI_COMM_WORLD);
    }

    // -----------------------------------------------------------------------
    // Decode on peer shard.
    // -----------------------------------------------------------------------
    MemRef<long long, 1> cachePosition({1}, 0LL);
    MemRef<float, 3> subResultContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 3> sub3DContainerDecode({1, 1, HiddenSize0});
    MemRef<int8_t, 4> mhaMemRef4DDecode({1, 1, 1, MaxTokenLength});
    MemRef<float, 3> mhaMemRef3D1Decode({1, 1, HiddenSize});
    MemRef<float, 3> mhaMemRef3D2Decode({1, 1, HiddenSize});
    MemRef<float, 2> tmp2DContainerDecode({1, HiddenSize0});
    MemRef<float, 2> sub2DContainerDecode({1, HiddenSize0});
    MemRef<float, 3> resultDecodeShard1({1, 1, VocabShardSize});
    int8_t *mhaMemRef4DPtrDecode = mhaMemRef4DDecode.getData();
    float *mhaMemRef3D1PtrDecode = mhaMemRef3D1Decode.getData();
    float *mhaMemRef3D2PtrDecode = mhaMemRef3D2Decode.getData();
    float *mhaOutputPtrDecode = tmp2DContainerDecode.getData();
    MemRef<long long, 1> cachePositionOutDecode({1}, 0LL);
    MemRefContainerDecode2 kvDecodeContainer0(cachePositionOutDecode, kv0[0],
                                              kv0[1], tmp2DContainerDecode);
    MemRefContainerDecode2 *kvDecodeContainerPtr0 = &kvDecodeContainer0;

    MPI_Request recv_req_decode[5];
    int ctrl = 0;

    for (int i = 1; i <= generateLen; i++) {
      MPI_Recv(&ctrl, 1, MPI_INT, source, 99, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      if (ctrl == 1) {
        break;
      }

      MPI_Irecv(subResultContainerDecode.getData(), HiddenSize0, MPI_FLOAT,
                source, 0, MPI_COMM_WORLD, &recv_req_decode[0]);
      MPI_Irecv(mhaMemRef4DPtrDecode, MaxTokenLength, MPI_INT8_T, source, 1,
                MPI_COMM_WORLD, &recv_req_decode[1]);
      MPI_Irecv(mhaMemRef3D1PtrDecode, HiddenSize, MPI_FLOAT, source, 2,
                MPI_COMM_WORLD, &recv_req_decode[2]);
      MPI_Irecv(mhaMemRef3D2PtrDecode, HiddenSize, MPI_FLOAT, source, 3,
                MPI_COMM_WORLD, &recv_req_decode[3]);
      MPI_Irecv(cachePosition.getData(), 1, MPI_LONG_LONG, source, 4,
                MPI_COMM_WORLD, &recv_req_decode[4]);

      MPI_Wait(&recv_req_decode[0], MPI_STATUS_IGNORE);

      for (int m = 0; m < times; m++) {
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS[m],
                                     &subResultContainerDecode);

        if (m == 0) {
          MPI_Waitall(4, &recv_req_decode[1], MPI_STATUSES_IGNORE);
        }

        _mlir_ciface_forward_decode2(
            kvDecodeContainerPtr0, &paramsContainersMHA[m], &cachePosition,
            &kv0[2 * m], &kv0[2 * m + 1], &mhaMemRef4DDecode,
            &mhaMemRef3D1Decode, &mhaMemRef3D2Decode, &sub3DContainerDecode);
        kv0[2 * m] = kvDecodeContainerPtr0->kcache;
        kv0[2 * m + 1] = kvDecodeContainerPtr0->vcache;
        tmp2DContainerDecode = kvDecodeContainerPtr0->data;
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        if (comm_sub != MPI_COMM_NULL) {
          reduceSumTwoRanksSendrecv(mhaOutputPtrDecode,
                                    sub2DContainerDecode.getData(), HiddenSize0,
                                    FrontendRank, 1000 + m, comm_sub);
        } else {
          std::memcpy(sub2DContainerDecode.getData(), mhaOutputPtrDecode,
                      sizeof(float) * HiddenSize0);
        }

        _mlir_ciface_forward_decode3(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);

        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS0[m],
                                     &subResultContainerDecode);

        _mlir_ciface_forward_decode5(&tmp2DContainerDecode,
                                     &paramsContainersMLP[m],
                                     &sub3DContainerDecode);
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        if (comm_sub != MPI_COMM_NULL) {
          reduceSumTwoRanksSendrecv(mhaOutputPtrDecode,
                                    sub2DContainerDecode.getData(), HiddenSize0,
                                    FrontendRank, 2000 + m, comm_sub);
        } else {
          std::memcpy(sub2DContainerDecode.getData(), mhaOutputPtrDecode,
                      sizeof(float) * HiddenSize0);
        }

        _mlir_ciface_forward_decode3(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
      }

      _mlir_ciface_forward_decode169(&resultDecodeShard1, &paramsContainer2,
                                     &subResultContainerDecode);

      const float *decodeShard1StartPtr = resultDecodeShard1.getData();
      const float *decodeShard1EndPtr = decodeShard1StartPtr + VocabShardSize;
      int localDecodeMaxIndex =
          findMaxIndex(decodeShard1StartPtr, decodeShard1EndPtr);
      float localDecodeMaxValue = decodeShard1StartPtr[localDecodeMaxIndex];

      (void)reduceMaxLoc(localDecodeMaxValue,
                         static_cast<int>(VocabShardSize) + localDecodeMaxIndex,
                         comm_sub);
    }

    if (comm_sub != MPI_COMM_NULL) {
      MPI_Comm_free(&comm_sub);
    }
    if (sub_group != MPI_GROUP_NULL) {
      MPI_Group_free(&sub_group);
    }
    if (world_group != MPI_GROUP_NULL) {
      MPI_Group_free(&world_group);
    }
  }

  MPI_Finalize();

  return 0;
}
