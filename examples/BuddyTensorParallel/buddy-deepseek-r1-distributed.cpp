#include <algorithm>
#include <array>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <string>
#include <vector>

using namespace buddy;
double total_time = 0;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;
constexpr size_t SubMaxTokenLength = 512;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HiddenSize0 = 1536;
constexpr size_t HeadNum = 2;
constexpr int FrontendRank = 0;
constexpr int PeerRank = 1;

namespace CommTag {
constexpr int PrefillHidden = 100;
constexpr int PrefillAuxMask = 101;
constexpr int PrefillAuxCos = 102;
constexpr int PrefillAuxSin = 103;
constexpr int PrefillReturnHidden = 104;
constexpr int DecodePacketA = 200;
constexpr int DecodePacketB = 201;
} // namespace CommTag

struct DecodePacketA {
  int ctrl = 0;
  long long cachePosition = 0LL;
  std::array<float, HiddenSize0> hidden;
};

struct DecodePacketB {
  std::array<int8_t, MaxTokenLength> mask;
  std::array<float, HiddenSize> cos;
  std::array<float, HiddenSize> sin;
};

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
struct MemRefContainer2temp {
  MemRef<float, 4> qcache;
  MemRef<float, 4> kcache;
  MemRef<float, 4> vcache;

  MemRefContainer2temp(MemRef<float, 4> m1, MemRef<float, 4> m2,
                   MemRef<float, 4> m3)
      : qcache(m1), kcache(m2), vcache(m3) {}
};

void packDecodePacketA(DecodePacketA &packet, int ctrl, long long cachePosition,
                       const float *hidden) {
  packet.ctrl = ctrl;
  packet.cachePosition = cachePosition;
  std::memcpy(packet.hidden.data(), hidden, sizeof(float) * HiddenSize0);
}

void unpackDecodePacketA(const DecodePacketA &packet, MemRef<float, 3> &hidden,
                         MemRef<long long, 1> &cachePosition) {
  std::memcpy(hidden.getData(), packet.hidden.data(), sizeof(float) * HiddenSize0);
  cachePosition.getData()[0] = packet.cachePosition;
}

void packDecodePacketB(DecodePacketB &packet, MemRefContainer0 &src) {
  std::memcpy(packet.mask.data(), src.mask.getData(), sizeof(int8_t) * MaxTokenLength);
  std::memcpy(packet.cos.data(), src.cos.getData(), sizeof(float) * HiddenSize);
  std::memcpy(packet.sin.data(), src.sin.getData(), sizeof(float) * HiddenSize);
}

void unpackDecodePacketB(const DecodePacketB &packet, MemRef<int8_t, 4> &mask,
                         MemRef<float, 3> &cos, MemRef<float, 3> &sin) {
  std::memcpy(mask.getData(), packet.mask.data(), sizeof(int8_t) * MaxTokenLength);
  std::memcpy(cos.getData(), packet.cos.data(), sizeof(float) * HiddenSize);
  std::memcpy(sin.getData(), packet.sin.data(), sizeof(float) * HiddenSize);
}

/// Declare DeepSeekR1 forward function.
extern "C" {
void _mlir_ciface_forward_prefill0(MemRefContainer0 *, MemRef<float, 1> *,
                                   Text<size_t, 2> *);
void _mlir_ciface_forward_prefill1(MemRef<float, 3> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill2(MemRefContainer2temp *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill3(MemRefContainer2 *, MemRef<float, 1> *,
                                   MemRef<int8_t, 4> *, MemRef<float, 3> *,
                                   MemRef<float, 3> *, MemRef<float, 4> *,
                                  MemRef<float, 4> *, MemRef<float, 4> *);

void _mlir_ciface_forward_prefill4(MemRef<float, 3> *, MemRef<float, 3> *,
                                   MemRef<float, 2> *);
void _mlir_ciface_forward_prefill6(MemRef<float, 2> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill197(MemRef<float, 3> *, MemRef<float, 1> *,
                                     MemRef<float, 3> *);

void _mlir_ciface_forward_decode0(MemRefContainer0 *, MemRef<float, 1> *,
                                  MemRef<long long, 2> *,
                                  MemRef<long long, 1> *);
void _mlir_ciface_forward_decode1(MemRef<float, 3> *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode2(MemRefContainer2temp *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode3(MemRefContainer2 *, MemRef<float, 1> *,
                                  MemRef<long long, 1> *, MemRef<float, 4> *,
                                  MemRef<float, 4> *, MemRef<int8_t, 4> *,
                                  MemRef<float, 3> *, MemRef<float, 3> *,
                                MemRef<float, 4> *, MemRef<float, 4> *, MemRef<float, 4> *);
void _mlir_ciface_forward_decode4(MemRef<float, 3> *, MemRef<float, 3> *,
                                  MemRef<float, 2> *);
void _mlir_ciface_forward_decode6(MemRef<float, 2> *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode197(MemRef<float, 3> *, MemRef<float, 1> *,
                                    MemRef<float, 3> *);
}

using HighResClock = std::chrono::high_resolution_clock;

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
  const auto tokenizeStart = HighResClock::now();
  inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
  const auto tokenizeEnd = HighResClock::now();
  const std::chrono::duration<double, std::milli> tokenizeTime =
      tokenizeEnd - tokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << tokenizeTime.count() << "ms"
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
                 sizeof(float) * params.getSize());
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

// -----------------------------------------------------------------------------
// DeepSeekR1 Inference Main Entry
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {

  /// Define directories of vacabulary and parameter file.
  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "/vocab.txt";

  // Common variables needed by all ranks
  int subSize = SubMaxTokenLength * HiddenSize0;
  int offset0 = subSize;
  int offset1 = subSize * 2;

  int rank, size;
  int generateLen = MaxTokenLength;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == FrontendRank) {
    /// Print the title of this example.
    const std::string title = "DeepSeekR1  Inference Powered by Buddy Compiler";
    std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

    // -----------------------------------------------------------------------
    // Frontend resources originally on rank0.
    // -----------------------------------------------------------------------
    Text<size_t, 2> outputContainer;
    outputContainer.loadVocab(vocabDir);

    MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize0});
    MemRef<int8_t, 4> myMemRef2({1, 1, MaxTokenLength, MaxTokenLength});
    MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize});
    MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize});
    MemRefContainer0 resultContainer(myMemRef1, myMemRef2, myMemRef3,
                                     myMemRef4);
    MemRefContainer0 *resultContainerPtr = &resultContainer;

    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 3> resultPrefill({1, MaxTokenLength, MaxVocabSize});
    MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
    MemRef<long long, 1> cachePosition({1}, 0LL);

    MemRef<float, 3> myMemRef_decode1({1, 1, HiddenSize0});
    MemRef<int8_t, 4> myMemRef_decode2({1, 1, 1, MaxTokenLength});
    MemRef<float, 3> myMemRef_decode3({1, 1, HiddenSize});
    MemRef<float, 3> myMemRef_decode4({1, 1, HiddenSize});
    MemRefContainer0 resultContainerDecode(myMemRef_decode1, myMemRef_decode2,
                                           myMemRef_decode3, myMemRef_decode4);
    MemRefContainer0 *resultContainerDecodePtr = &resultContainerDecode;

    MemRef<float, 3> resultDecode({1, 1, MaxVocabSize});

    // Load embedding / unembedding parameters on rank1.
    constexpr size_t param_size0 = 233373760;
    const std::string paramsDir0 =
        deepSeekR1BuildDir + "/subgraph0_prefill0_arg0.data";
    constexpr size_t param_size1 = 233375232;
    const std::string paramsDir1 =
        deepSeekR1BuildDir + "/subgraph0_prefill197_arg0.data";
    MemRef<float, 1> paramsContainer0({param_size0});
    loadParameters(paramsDir0, paramsContainer0);
    MemRef<float, 1> paramsContainer1({param_size1});
    loadParameters(paramsDir1, paramsContainer1);

    // -----------------------------------------------------------------------
    // Worker resources originally on rank1.
    // -----------------------------------------------------------------------
    MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize0});
    MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize0});
    std::vector<MemRef<float, 4>> kv0;
    kv0.reserve(56);
    for (int i = 0; i < 56; ++i) {
      kv0.emplace_back(std::vector<size_t>{1, 1, MaxTokenLength, HiddenSize});
    }
    MemRefContainer2 kvContainer0(kv0[0], kv0[1], tmp2DContainer);
    MemRefContainer2 *kvContainerPtr0 = &kvContainer0;

    MemRef<float, 4> tempQ4D({1, 6, MaxTokenLength, HiddenSize});
    MemRef<float, 4> tempK4D({1, 1, MaxTokenLength, HiddenSize});
    MemRef<float, 4> tempV4D({1, 1, MaxTokenLength, HiddenSize});
    MemRefContainer2temp kvContainerTemp(tempQ4D, tempK4D, tempV4D);
    MemRefContainer2temp *kvContainerTempPtr = &kvContainerTemp;

    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();

    constexpr size_t paramSizeRMS = 1536;
    constexpr size_t paramSizeMHA0 = 1573888;
    constexpr size_t paramSizeMHA1 = 1179648;
    constexpr size_t paramSizeMLP = 20643840;
    int times = 28;
    int peerRank = PeerRank;

    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA0, paramsDirsMHA1, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA0, paramsContainersMHA1,
        paramsContainersMLP;

    for (int i = 1; i < 197; i += 7) {
      paramsDirsRMS.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0.data");
      paramsDirsRMS0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 4) + "_arg0.data");
    }
    for (int i = 2; i < 197; i += 7) {
      paramsDirsMHA0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i) + "_arg0.data");
      paramsDirsMHA1.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 1) + "_arg0.data");
    }
    for (int i = 6; i < 197; i += 7) {
      paramsDirsMLP.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0.data");
    }

    for (int i = 0; i < times; i++) {
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);

      MemRef<float, 1> paramsContainerMHA0({paramSizeMHA0});
      loadParameters(paramsDirsMHA0[i], paramsContainerMHA0);
      paramsContainersMHA0.push_back(paramsContainerMHA0);

      MemRef<float, 1> paramsContainerMHA1({paramSizeMHA1});
      loadParameters(paramsDirsMHA1[i], paramsContainerMHA1);
      paramsContainersMHA1.push_back(paramsContainerMHA1);

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

    std::string inputStr;
    getUserInput(inputStr);
    Text<size_t, 2> inputContainerPrefill(inputStr);
    tokenizeInput(vocabDir, inputContainerPrefill);
    std::cout << "Input token count: " << inputContainerPrefill.getTokenCnt()
              << std::endl;

    MPI_Request send_req[2], recv_req[2];
    MPI_Request send_req_mha[3];
    DecodePacketA decodePacketASend;
    DecodePacketB decodePacketBSend;

    float *inputPtr = nullptr;
    float *outputPtr = tmp3DMemRef.getData();
    float *outputPtrDecode = myMemRef_decode1.getData();
    (void)outputPtrDecode;

    double prefillTokensPerSec = 0.0;
    double prefillTokensPerSecInput = 0.0;
    const auto inferenceStart = HighResClock::now();
    std::cout << "\n\033[33;1m[Inference Start]\033[0m" << std::endl;

    auto t0 = HighResClock::now();
    _mlir_ciface_forward_prefill0(resultContainerPtr, &paramsContainer0,
                                  &inputContainerPrefill);
    inputPtr = resultContainerPtr->data.getData();

    t0 = HighResClock::now();
    std::memcpy(subResultPtr, inputPtr, sizeof(float) * subSize);

    t0 = HighResClock::now();
    MPI_Isend(inputPtr + offset0, subSize, MPI_FLOAT, peerRank,
              CommTag::PrefillHidden, MPI_COMM_WORLD, &send_req[0]);

    t0 = HighResClock::now();
    MPI_Isend(resultContainerPtr->mask.getData(),
              MaxTokenLength * MaxTokenLength, MPI_BYTE, peerRank,
              CommTag::PrefillAuxMask, MPI_COMM_WORLD, &send_req_mha[0]);
    MPI_Isend(resultContainerPtr->cos.getData(),
              MaxTokenLength * HiddenSize, MPI_FLOAT, peerRank,
              CommTag::PrefillAuxCos, MPI_COMM_WORLD, &send_req_mha[1]);
    MPI_Isend(resultContainerPtr->sin.getData(),
              MaxTokenLength * HiddenSize, MPI_FLOAT, peerRank,
              CommTag::PrefillAuxSin, MPI_COMM_WORLD, &send_req_mha[2]);

    t0 = HighResClock::now();
    MPI_Irecv(outputPtr + offset0, subSize, MPI_FLOAT, peerRank,
              CommTag::PrefillReturnHidden, MPI_COMM_WORLD, &recv_req[0]);

    for (int m = 0; m < times; m++) {
      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS[m],
                                    &subResultContainer);
      rmsPtr = sub3DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = HighResClock::now();
        MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                      subSize, MPI_FLOAT, comm_sub);
      }

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill2(kvContainerTempPtr, &paramsContainersMHA0[m],
                                    &tmp3DMemRef);

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill3(kvContainerPtr0, &paramsContainersMHA1[m],
                                    &resultContainerPtr->mask,
                                    &resultContainerPtr->cos,
                                    &resultContainerPtr->sin,
                                    &kvContainerTempPtr->qcache,
                                    &kvContainerTempPtr->kcache,
                                    &kvContainerTempPtr->vcache);

      kv0[2 * m] = kvContainerPtr0->kcache;
      kv0[2 * m + 1] = kvContainerPtr0->vcache;
      tmp2DContainer = kvContainerPtr0->data;
      mhaOutputPtr = tmp2DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = HighResClock::now();
        MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                 MPI_SUM, comm_sub);
      }

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill4(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS0[m],
                                    &subResultContainer);
      rmsPtr = sub3DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = HighResClock::now();
        MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                      subSize, MPI_FLOAT, comm_sub);
      }

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill6(&tmp2DContainer, &paramsContainersMLP[m],
                                    &tmp3DMemRef);
      mhaOutputPtr = tmp2DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = HighResClock::now();
        MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                 MPI_SUM, comm_sub);
      }

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill4(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);
    }

    t0 = HighResClock::now();
    std::memcpy(outputPtr, subResultContainer.getData(), sizeof(float) * subSize);

    t0 = HighResClock::now();
    MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

    MPI_Request prefillSends[4] = {send_req[0], send_req_mha[0],
                                   send_req_mha[1], send_req_mha[2]};
    t0 = HighResClock::now();
    MPI_Waitall(4, prefillSends, MPI_STATUSES_IGNORE);

    t0 = HighResClock::now();
    _mlir_ciface_forward_prefill197(&resultPrefill, &paramsContainer1,
                                    &tmp3DMemRef);

    const auto inferenceEnd = HighResClock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

    int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
    const float *startPtr = resultPrefill.getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainerPrefill.getStr(maxIndex);
    printIterInfo(0, tok, inferenceTime.count() / 1000.0);

    const double prefillSeconds = inferenceTime.count() / 1000.0;
    if (prefillSeconds > 0.0) {
      prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
      prefillTokensPerSecInput =
          static_cast<double>(inputContainerPrefill.getTokenCnt()) /
          prefillSeconds;
    }

    inputContainerDecode.getData()[0] = (long long)maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
    generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
    double decodeTimeAccumMs = 0.0;
    size_t decodeTokens = 0;

    MPI_Request send_req_decode[2];
    bool sentStop = false;

    // -----------------------------------------------------------------------
    // Local decode worker resources (rank1 shard).
    // Keep a worker-local snapshot of decode inputs so the merged frontend path
    // still follows the original worker dataflow.
    // -----------------------------------------------------------------------
    MemRef<float, 3> subResultContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 3> sub3DContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 2> tmp2DContainerDecode({1, HiddenSize0});
    MemRef<float, 2> sub2DContainerDecode({1, HiddenSize0});
    // Do not cache subResultContainerDecode.getData() across iterations.
    // MLIR iface calls may update the MemRef descriptor, so a cached raw pointer
    // can become stale and point to the previous iteration's buffer.
    float *mhaOutputPtrDecode = tmp2DContainerDecode.getData();
    float *sub2DPtrDecode = sub2DContainerDecode.getData();
    (void)sub2DPtrDecode;

    MemRefContainer2 kvDecodeContainer0(kv0[0], kv0[1], tmp2DContainerDecode);
    MemRefContainer2 *kvDecodeContainerPtr0 = &kvDecodeContainer0;

    MemRef<float, 4> tempDecodeQ4D({1, 6, 1, HiddenSize});
    MemRef<float, 4> tempDecodeK4D({1, 1, 1, HiddenSize});
    MemRef<float, 4> tempDecodeV4D({1, 1, 1, HiddenSize});
    MemRefContainer2temp kvDecodeContainerTemp(tempDecodeQ4D, tempDecodeK4D,
                                               tempDecodeV4D);
    MemRefContainer2temp *kvDecodeContainerTempPtr = &kvDecodeContainerTemp;

    for (int i = 1; i <= generateLen; i++) {
      const auto decodeIterStart = HighResClock::now();
      t0 = HighResClock::now();
      _mlir_ciface_forward_decode0(resultContainerDecodePtr, &paramsContainer0,
                                   &inputContainerDecode, &cachePosition);
      inputPtr = resultContainerDecodePtr->data.getData();

      t0 = HighResClock::now();
      std::memcpy(subResultContainerDecode.getData(), inputPtr,
                  sizeof(float) * HiddenSize0);
      packDecodePacketA(decodePacketASend, 0, cachePosition.getData()[0],
                        inputPtr);
      packDecodePacketB(decodePacketBSend, *resultContainerDecodePtr);

      t0 = HighResClock::now();
      MPI_Isend(reinterpret_cast<const void *>(&decodePacketASend),
                static_cast<int>(sizeof(DecodePacketA)), MPI_BYTE, peerRank,
                CommTag::DecodePacketA, MPI_COMM_WORLD, &send_req_decode[0]);
      MPI_Isend(reinterpret_cast<const void *>(&decodePacketBSend),
                static_cast<int>(sizeof(DecodePacketB)), MPI_BYTE, peerRank,
                CommTag::DecodePacketB, MPI_COMM_WORLD, &send_req_decode[1]);

      for (int m = 0; m < times; m++) {
        t0 = HighResClock::now();
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS[m],
                                     &subResultContainerDecode);

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode2(kvDecodeContainerTempPtr,
                                     &paramsContainersMHA0[m],
                                     &sub3DContainerDecode);

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode3(
            kvDecodeContainerPtr0, &paramsContainersMHA1[m],
            &cachePosition, &kv0[2 * m], &kv0[2 * m + 1],
            &resultContainerDecodePtr->mask, &resultContainerDecodePtr->cos,
            &resultContainerDecodePtr->sin, &kvDecodeContainerTempPtr->qcache,
            &kvDecodeContainerTempPtr->kcache,
            &kvDecodeContainerTempPtr->vcache);

        kv0[2 * m] = kvDecodeContainerPtr0->kcache;
        kv0[2 * m + 1] = kvDecodeContainerPtr0->vcache;
        tmp2DContainerDecode = kvDecodeContainerPtr0->data;
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        if (comm_sub != MPI_COMM_NULL) {
          t0 = HighResClock::now();
          MPI_Allreduce(mhaOutputPtrDecode, sub2DContainerDecode.getData(),
                        HiddenSize0, MPI_FLOAT, MPI_SUM, comm_sub);
        }

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode4(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS0[m],
                                     &subResultContainerDecode);

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode6(&tmp2DContainerDecode,
                                     &paramsContainersMLP[m],
                                     &sub3DContainerDecode);
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();

        if (comm_sub != MPI_COMM_NULL) {
          t0 = HighResClock::now();
          MPI_Allreduce(mhaOutputPtrDecode, sub2DContainerDecode.getData(),
                        HiddenSize0, MPI_FLOAT, MPI_SUM, comm_sub);
        }

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode4(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);

      }

      t0 = HighResClock::now();
      MPI_Waitall(2, send_req_decode, MPI_STATUSES_IGNORE);

      t0 = HighResClock::now();
      _mlir_ciface_forward_decode197(&resultDecode, &paramsContainer1,
                                     &subResultContainerDecode);

      const auto decodeIterEnd = HighResClock::now();
      const std::chrono::duration<double, std::milli> decodeIterTime =
          decodeIterEnd - decodeIterStart;
      decodeTimeAccumMs += decodeIterTime.count();
      decodeTokens += 1;

      const float *decodeStartPtr = resultDecode.getData();
      const float *decodeEndPtr = decodeStartPtr + MaxVocabSize;
      int maxIndex = findMaxIndex(decodeStartPtr, decodeEndPtr);
      std::string tok = inputContainerPrefill.getStr(maxIndex);
      printIterInfo(i, tok, decodeIterTime.count() / 1000.0);

      if (maxIndex == 151643) {
        packDecodePacketA(decodePacketASend, 1, cachePosition.getData()[0],
                          subResultContainerDecode.getData());
        t0 = HighResClock::now();
        MPI_Send(reinterpret_cast<const void *>(&decodePacketASend),
                 static_cast<int>(sizeof(DecodePacketA)), MPI_BYTE, peerRank,
                 CommTag::DecodePacketA, MPI_COMM_WORLD);
        sentStop = true;
        break;
      }

      inputContainerDecode.getData()[0] = maxIndex;
      outputContainer.appendTokenIdx(maxIndex);
      cachePosition.getData()[0] += 1;
    }

    if (!sentStop) {
      packDecodePacketA(decodePacketASend, 1, cachePosition.getData()[0],
                        subResultContainerDecode.getData());
      t0 = HighResClock::now();
      MPI_Send(reinterpret_cast<const void *>(&decodePacketASend),
               static_cast<int>(sizeof(DecodePacketA)), MPI_BYTE, peerRank,
               CommTag::DecodePacketA, MPI_COMM_WORLD);
    }

    double decodeSeconds = decodeTimeAccumMs / 1000.0;
    const double decodeTokensPerSec =
        decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
                            : 0.0;
    const double decodeAvgLatencyMs =
        decodeTokens > 0
            ? decodeSeconds * 1000.0 / static_cast<double>(decodeTokens)
            : 0.0;
    std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time
              << std::endl;
    std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
              << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Prefilling-real]\033[0m "
              << prefillTokensPerSecInput << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Prefill latency]\033[0m " << prefillSeconds
              << " s" << std::endl;
    std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTokensPerSec
              << " tokens/s" << std::endl;
    std::cout << "\033[33;1m[Decode avg latency]\033[0m "
              << decodeAvgLatencyMs << " ms/token" << std::endl;
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

    // === Worker shard on rank2 ===
    MemRef<float, 3> subResultContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<float, 3> sub3DContainer({1, SubMaxTokenLength, HiddenSize0});
    MemRef<int8_t, 4> mhaMemRef4D({1, 1, MaxTokenLength, MaxTokenLength});
    MemRef<float, 3> mhaMemRef3D1({1, MaxTokenLength, HiddenSize});
    MemRef<float, 3> mhaMemRef3D2({1, MaxTokenLength, HiddenSize});
    MemRef<float, 3> tmp3DMemRef({1, MaxTokenLength, HiddenSize0});
    MemRef<float, 2> tmp2DContainer({MaxTokenLength, HiddenSize0});
    MemRef<float, 2> sub2DContainer({SubMaxTokenLength, HiddenSize0});
    std::vector<MemRef<float, 4>> kv0;
    kv0.reserve(56);
    for (int i = 0; i < 56; ++i) {
      kv0.emplace_back(std::vector<size_t>{1, 1, MaxTokenLength, HiddenSize});
    }
    MemRefContainer2 kvContainer0(kv0[0], kv0[1], tmp2DContainer);
    MemRefContainer2 *kvContainerPtr0 = &kvContainer0;

    MemRef<float, 4> tempQ4D({1, 6, MaxTokenLength, HiddenSize});
    MemRef<float, 4> tempK4D({1, 1, MaxTokenLength, HiddenSize});
    MemRef<float, 4> tempV4D({1, 1, MaxTokenLength, HiddenSize});
    MemRefContainer2temp kvContainerTemp(tempQ4D, tempK4D, tempV4D);
    MemRefContainer2temp *kvContainerTempPtr = &kvContainerTemp;

    float *subResultPtr = subResultContainer.getData();
    float *rmsPtr = sub3DContainer.getData();
    float *mhaOutputPtr = tmp2DContainer.getData();
    float *sub2DPtr = sub2DContainer.getData();

    constexpr size_t paramSizeRMS = 1536;
    constexpr size_t paramSizeMHA0 = 1573888;
    constexpr size_t paramSizeMHA1 = 1179648;
    constexpr size_t paramSizeMLP = 20643840;
    int times = 28;
    int source = FrontendRank;
    int nextRank = FrontendRank;
    MPI_Request recv_req[2];
    MPI_Request mha_recv_req[3];
    MPI_Request decode_recv_packet_b = MPI_REQUEST_NULL;
    DecodePacketA decodePacketARecv;
    DecodePacketB decodePacketBRecv;
    std::vector<std::string> paramsDirsRMS, paramsDirsRMS0;
    std::vector<std::string> paramsDirsMHA0, paramsDirsMHA1, paramsDirsMLP;
    std::vector<MemRef<float, 1>> paramsContainersRMS, paramsContainersRMS0;
    std::vector<MemRef<float, 1>> paramsContainersMHA0, paramsContainersMHA1,
        paramsContainersMLP;

    for (int i = 1; i < 197; i += 7) {
      paramsDirsRMS.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg0.data");
      paramsDirsRMS0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 4) + "_arg0.data");
    }
    for (int i = 2; i < 197; i += 7) {
      paramsDirsMHA0.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i) + "_arg1.data");
      paramsDirsMHA1.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                  std::to_string(i + 1) + "_arg1.data");
    }
    for (int i = 6; i < 197; i += 7) {
      paramsDirsMLP.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                 std::to_string(i) + "_arg1.data");
    }

    for (int i = 0; i < times; i++) {
      MemRef<float, 1> paramsContainerRMS({paramSizeRMS});
      loadParameters(paramsDirsRMS[i], paramsContainerRMS);
      paramsContainersRMS.push_back(paramsContainerRMS);

      MemRef<float, 1> paramsContainerMHA0({paramSizeMHA0});
      loadParameters(paramsDirsMHA0[i], paramsContainerMHA0);
      paramsContainersMHA0.push_back(paramsContainerMHA0);

      MemRef<float, 1> paramsContainerMHA1({paramSizeMHA1});
      loadParameters(paramsDirsMHA1[i], paramsContainerMHA1);
      paramsContainersMHA1.push_back(paramsContainerMHA1);

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

    auto t0 = HighResClock::now();
    MPI_Irecv(mhaMemRef4D.getData(), MaxTokenLength * MaxTokenLength, MPI_BYTE,
              source, CommTag::PrefillAuxMask, MPI_COMM_WORLD,
              &mha_recv_req[0]);
    MPI_Irecv(mhaMemRef3D1.getData(), MaxTokenLength * HiddenSize, MPI_FLOAT,
              source, CommTag::PrefillAuxCos, MPI_COMM_WORLD,
              &mha_recv_req[1]);
    MPI_Irecv(mhaMemRef3D2.getData(), MaxTokenLength * HiddenSize, MPI_FLOAT,
              source, CommTag::PrefillAuxSin, MPI_COMM_WORLD,
              &mha_recv_req[2]);

    t0 = HighResClock::now();
    MPI_Irecv(subResultPtr, subSize, MPI_FLOAT, source, CommTag::PrefillHidden,
              MPI_COMM_WORLD, &recv_req[0]);

    t0 = HighResClock::now();
    MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

    for (int m = 0; m < times; m++) {
      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS[m],
                                    &subResultContainer);
      rmsPtr = sub3DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = HighResClock::now();
        MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                      subSize, MPI_FLOAT, comm_sub);
      }

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill2(kvContainerTempPtr, &paramsContainersMHA0[m],
                                    &tmp3DMemRef);

      if (m == 0) {
        t0 = HighResClock::now();
        MPI_Waitall(3, mha_recv_req, MPI_STATUSES_IGNORE);
      }

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill3(kvContainerPtr0, &paramsContainersMHA1[m],
                                    &mhaMemRef4D, &mhaMemRef3D1, &mhaMemRef3D2,
                                    &kvContainerTempPtr->qcache,
                                    &kvContainerTempPtr->kcache,
                                    &kvContainerTempPtr->vcache);

      kv0[2 * m] = kvContainerPtr0->kcache;
      kv0[2 * m + 1] = kvContainerPtr0->vcache;
      tmp2DContainer = kvContainerPtr0->data;
      mhaOutputPtr = tmp2DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = HighResClock::now();
        MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                 MPI_SUM, comm_sub);
      }

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill4(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill1(&sub3DContainer, &paramsContainersRMS0[m],
                                    &subResultContainer);
      rmsPtr = sub3DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = HighResClock::now();
        MPI_Allgather(rmsPtr, subSize, MPI_FLOAT, tmp3DMemRef.getData(),
                      subSize, MPI_FLOAT, comm_sub);
      }

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill6(&tmp2DContainer, &paramsContainersMLP[m],
                                    &tmp3DMemRef);
      mhaOutputPtr = tmp2DContainer.getData();

      if (comm_sub != MPI_COMM_NULL) {
        t0 = HighResClock::now();
        MPI_Reduce_scatter_block(mhaOutputPtr, sub2DPtr, subSize, MPI_FLOAT,
                                 MPI_SUM, comm_sub);
      }

      t0 = HighResClock::now();
      _mlir_ciface_forward_prefill4(&subResultContainer, &subResultContainer,
                                    &sub2DContainer);

      if (m == (times - 1)) {
        subResultPtr = subResultContainer.getData();
        t0 = HighResClock::now();
        MPI_Send(subResultPtr, subSize, MPI_FLOAT, nextRank,
                 CommTag::PrefillReturnHidden, MPI_COMM_WORLD);
      }
    }

    // decode
    MemRef<long long, 1> cachePosition({1}, 0LL);
    MemRef<float, 3> subResultContainerDecode({1, 1, HiddenSize0});
    MemRef<float, 3> sub3DContainerDecode({1, 1, HiddenSize0});
    MemRef<int8_t, 4> mhaMemRef4DDecode({1, 1, 1, MaxTokenLength});
    MemRef<float, 3> mhaMemRef3D1Decode({1, 1, HiddenSize});
    MemRef<float, 3> mhaMemRef3D2Decode({1, 1, HiddenSize});
    MemRef<float, 2> tmp2DContainerDecode({1, HiddenSize0});
    MemRef<float, 2> sub2DContainerDecode({1, HiddenSize0});
    // Do not cache subResultContainerDecode.getData() across iterations.
    // The active MemRef descriptor may change after MLIR iface calls.
    float *mhaOutputPtrDecode = tmp2DContainerDecode.getData();

    MemRefContainer2 kvDecodeContainer0(kv0[0], kv0[1], tmp2DContainerDecode);
    MemRefContainer2 *kvDecodeContainerPtr0 = &kvDecodeContainer0;

    MemRef<float, 4> tempDecodeQ4D({1, 6, 1, HiddenSize});
    MemRef<float, 4> tempDecodeK4D({1, 1, 1, HiddenSize});
    MemRef<float, 4> tempDecodeV4D({1, 1, 1, HiddenSize});
    MemRefContainer2temp kvDecodeContainerTemp(tempDecodeQ4D, tempDecodeK4D,
                                               tempDecodeV4D);
    MemRefContainer2temp *kvDecodeContainerTempPtr = &kvDecodeContainerTemp;
    for (int i = 1; i <= generateLen; i++) {
      t0 = HighResClock::now();
      MPI_Recv(reinterpret_cast<void *>(&decodePacketARecv),
               static_cast<int>(sizeof(DecodePacketA)), MPI_BYTE, source,
               CommTag::DecodePacketA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (decodePacketARecv.ctrl == 1) {
        break;
      }

      t0 = HighResClock::now();
      unpackDecodePacketA(decodePacketARecv, subResultContainerDecode,
                          cachePosition);

      t0 = HighResClock::now();
      MPI_Irecv(reinterpret_cast<void *>(&decodePacketBRecv),
                static_cast<int>(sizeof(DecodePacketB)), MPI_BYTE, source,
                CommTag::DecodePacketB, MPI_COMM_WORLD, &decode_recv_packet_b);

      for (int m = 0; m < times; m++) {
        t0 = HighResClock::now();
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS[m],
                                     &subResultContainerDecode);

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode2(kvDecodeContainerTempPtr,
                                     &paramsContainersMHA0[m],
                                     &sub3DContainerDecode);

        if (m == 0) {
          t0 = HighResClock::now();
          MPI_Wait(&decode_recv_packet_b, MPI_STATUS_IGNORE);

          t0 = HighResClock::now();
          unpackDecodePacketB(decodePacketBRecv, mhaMemRef4DDecode,
                              mhaMemRef3D1Decode, mhaMemRef3D2Decode);
        }

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode3(
            kvDecodeContainerPtr0, &paramsContainersMHA1[m], &cachePosition,
            &kv0[2 * m], &kv0[2 * m + 1], &mhaMemRef4DDecode,
            &mhaMemRef3D1Decode, &mhaMemRef3D2Decode,
            &kvDecodeContainerTempPtr->qcache,
            &kvDecodeContainerTempPtr->kcache,
            &kvDecodeContainerTempPtr->vcache);

        kv0[2 * m] = kvDecodeContainerPtr0->kcache;
        kv0[2 * m + 1] = kvDecodeContainerPtr0->vcache;
        tmp2DContainerDecode = kvDecodeContainerPtr0->data;
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();
        if (comm_sub != MPI_COMM_NULL) {
          t0 = HighResClock::now();
          MPI_Allreduce(mhaOutputPtrDecode, sub2DContainerDecode.getData(),
                        HiddenSize0, MPI_FLOAT, MPI_SUM, comm_sub);
        }

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode4(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode1(&sub3DContainerDecode,
                                     &paramsContainersRMS0[m],
                                     &subResultContainerDecode);

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode6(&tmp2DContainerDecode,
                                     &paramsContainersMLP[m],
                                     &sub3DContainerDecode);
        mhaOutputPtrDecode = tmp2DContainerDecode.getData();
        if (comm_sub != MPI_COMM_NULL) {
          t0 = HighResClock::now();
          MPI_Allreduce(mhaOutputPtrDecode, sub2DContainerDecode.getData(),
                        HiddenSize0, MPI_FLOAT, MPI_SUM, comm_sub);
        }

        t0 = HighResClock::now();
        _mlir_ciface_forward_decode4(&subResultContainerDecode,
                                     &subResultContainerDecode,
                                     &sub2DContainerDecode);
      }
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
