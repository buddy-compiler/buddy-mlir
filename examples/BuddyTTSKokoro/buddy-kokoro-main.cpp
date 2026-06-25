//===- buddy-kokoro-main.cpp ----------------------------------------------===//
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

#include <algorithm>
#include <buddy/Core/Container.h>
#include <buddy/DAP/AudioContainer.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/time.h>
#include <utility>

constexpr size_t KokoroSeqLen = 16;
constexpr size_t KokoroAlignLen = 65;
constexpr size_t KokoroPredictorParamsSize = 14464562;
constexpr size_t KokoroVocoderParamsSize = 67345480;
constexpr size_t KokoroRngStateSize = 1024;
constexpr size_t KokoroRefStyleSize = 256;
constexpr size_t KokoroStyleSize = 128;
constexpr size_t KokoroDurationHiddenWidth = 640;
constexpr size_t KokoroAudioSamples = 39000;
constexpr int KokoroSampleRate = 24000;

// The importer splits Kokoro at the data-dependent duration expansion.  The
// predictor graph produces token-level duration information and intermediate
// style/hidden tensors.  The C++ bridge turns predicted durations into a fixed
// frame-index tensor before calling the vocoder graph.
struct PredictorReturns {
  MemRef<long long, 1> length;
  MemRef<bool, 2> textMask;
  MemRef<float, 2> style;
  MemRef<float, 3> durationHidden;
  MemRef<long long, 1> predDur;
  MemRef<long long, 1> tokenPositions;
};

extern "C" double _mlir_ciface_rtclock() {
  struct timeval tp;
  int stat = gettimeofday(&tp, nullptr);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
}

extern "C" void _mlir_ciface_forward_predictor(PredictorReturns *result,
                                               MemRef<float, 1> *params,
                                               MemRef<long long, 1> *rngState,
                                               MemRef<long long, 2> *inputIds,
                                               MemRef<float, 2> *refStyle);

extern "C" void _mlir_ciface_forward_vocoder(
    MemRef<float, 1> *result, MemRef<float, 1> *params,
    MemRef<long long, 2> *inputIds, MemRef<long long, 1> *indices,
    MemRef<float, 3> *durationHidden, MemRef<float, 2> *style,
    MemRef<bool, 2> *textMask, MemRef<float, 2> *refStyle);

void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

void printStageLabel(const std::string &stage) {
  std::cout << "\033[32;1m[" << stage << "] \033[0m";
}

void printDone(double seconds) {
  std::cout << "\033[32;1mdone\033[0m in " << std::fixed << std::setprecision(3)
            << seconds << "s" << std::endl;
}

template <typename Func>
double timeSection(const std::string &stage, Func &&func) {
  printStageLabel(stage);
  std::cout << "running..." << std::flush;
  const auto start = std::chrono::high_resolution_clock::now();
  func();
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed = end - start;
  std::cout << "\r";
  printStageLabel(stage);
  printDone(elapsed.count());
  return elapsed.count();
}

template <typename T, size_t N>
void loadBinary(const std::string &path, MemRef<T, N> &memref) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("failed to open " + path);
  file.read(reinterpret_cast<char *>(memref.getData()),
            sizeof(T) * memref.getSize());
  if (file.gcount() !=
      static_cast<std::streamsize>(sizeof(T) * memref.getSize()))
    throw std::runtime_error("short read from " + path);
}

void saveWav(const std::string &path, MemRef<float, 1> &&audio) {
  dap::Audio<float, 1> wav(std::move(audio));
  wav.setBitDepth(16);
  wav.setChannelsNum(1);
  wav.setSampleRate(KokoroSampleRate);
  wav.setSamplesNum(wav.getSize());
  if (!wav.saveToFile(path, "wav"))
    throw std::runtime_error("failed to save wav to " + path);
}

void printAudioStats(MemRef<float, 1> &audio) {
  double sumSquares = 0.0;
  float minValue = 0.0f;
  float maxValue = 0.0f;
  float peak = 0.0f;
  if (audio.getSize() > 0) {
    minValue = audio.getData()[0];
    maxValue = audio.getData()[0];
  }
  for (size_t i = 0; i < audio.getSize(); ++i) {
    const float sample = audio.getData()[i];
    minValue = std::min(minValue, sample);
    maxValue = std::max(maxValue, sample);
    peak = std::max(peak, std::abs(sample));
    sumSquares += static_cast<double>(sample) * static_cast<double>(sample);
  }
  const double rms =
      audio.getSize() == 0 ? 0.0 : std::sqrt(sumSquares / audio.getSize());
  printLogLabel();
  std::cout << "Audio stats: min=" << minValue << ", max=" << maxValue
            << ", peak=" << peak << ", rms=" << rms << std::endl;
}

void fillDefaultRngState(MemRef<long long, 1> &rngState) {
  // The predictor graph exposes Buddy's int64 parameter pack as an input.  For
  // this static sample those values are deterministic runtime state rather
  // than model weights, so the driver recreates them locally.
  std::fill(rngState.getData(), rngState.getData() + rngState.getSize(), 0LL);
  for (size_t i = 0; i < 512 && i < rngState.getSize(); ++i)
    rngState.getData()[i] = static_cast<long long>(i);
}

size_t buildFrameIndices(PredictorReturns &ret, MemRef<long long, 1> &indices) {
  // Kokoro normally constructs an alignment matrix whose width depends on the
  // predicted durations.  The MLIR graphs are fixed-shape, so this bridge
  // materializes the equivalent frame-index vector and pads/truncates it to the
  // vocoder shape chosen during import.
  size_t cursor = 0;
  for (size_t token = 0; token < KokoroSeqLen && cursor < KokoroAlignLen;
       ++token) {
    long long duration = std::max<long long>(1, ret.predDur.getData()[token]);
    for (long long j = 0; j < duration && cursor < KokoroAlignLen; ++j)
      indices.getData()[cursor++] = static_cast<long long>(token);
  }
  while (cursor < KokoroAlignLen)
    indices.getData()[cursor++] = static_cast<long long>(KokoroSeqLen - 1);
  return cursor;
}

std::string optionValue(int argc, char **argv, const std::string &name,
                        const std::string &fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name)
      return argv[i + 1];
  }
  return fallback;
}

int main(int argc, char **argv) {
  const std::string title =
      "Kokoro Static TTS Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  const std::string buildDir =
#ifdef KOKORO_EXAMPLE_BUILD_PATH
      KOKORO_EXAMPLE_BUILD_PATH;
#else
      "./";
#endif

  const std::string params0Path =
      optionValue(argc, argv, "--params0", buildDir + "arg0_predictor.data");
  const std::string params1Path =
      optionValue(argc, argv, "--params1", buildDir + "arg0_vocoder.data");
  const std::string inputIdsPath =
      optionValue(argc, argv, "--tokens", buildDir + "input_ids.data");
  const std::string refStylePath =
      optionValue(argc, argv, "--ref-s", buildDir + "ref_s.data");
  const std::string outputPath =
      optionValue(argc, argv, "--output", buildDir + "kokoro_static.wav");

  try {
    printLogLabel();
    std::cout << "Example data dir: " << std::filesystem::absolute(buildDir)
              << std::endl;
    printLogLabel();
    std::cout << "Predictor params: " << std::filesystem::absolute(params0Path)
              << std::endl;
    printLogLabel();
    std::cout << "Vocoder params: " << std::filesystem::absolute(params1Path)
              << std::endl;
    printLogLabel();
    std::cout << "Token ids: " << std::filesystem::absolute(inputIdsPath)
              << std::endl;
    printLogLabel();
    std::cout << "Reference style: " << std::filesystem::absolute(refStylePath)
              << std::endl;
    printLogLabel();
    std::cout << "Static shapes: seq=" << KokoroSeqLen
              << ", align=" << KokoroAlignLen
              << ", audio_samples=" << KokoroAudioSamples
              << ", sample_rate=" << KokoroSampleRate << std::endl
              << std::endl;

    MemRef<float, 1> params0({KokoroPredictorParamsSize});
    MemRef<float, 1> params1({KokoroVocoderParamsSize});
    MemRef<long long, 1> rngState({KokoroRngStateSize}, 0LL);
    MemRef<long long, 2> inputIds({1, KokoroSeqLen}, 0LL);
    MemRef<float, 2> refStyle({1, KokoroRefStyleSize}, 0.0f);

    timeSection("Load data", [&]() {
      loadBinary(params0Path, params0);
      loadBinary(params1Path, params1);
      fillDefaultRngState(rngState);
      loadBinary(inputIdsPath, inputIds);
      loadBinary(refStylePath, refStyle);
    });

    PredictorReturns predictorRet = {
        MemRef<long long, 1>({1}, 0LL),
        MemRef<bool, 2>({1, KokoroSeqLen}, false),
        MemRef<float, 2>({1, KokoroStyleSize}, 0.0f),
        MemRef<float, 3>({1, KokoroSeqLen, KokoroDurationHiddenWidth}, 0.0f),
        MemRef<long long, 1>({KokoroSeqLen}, 1LL),
        MemRef<long long, 1>({KokoroSeqLen}, 0LL)};
    MemRef<long long, 1> frameIndices({KokoroAlignLen}, 0LL);
    MemRef<float, 1> audio({KokoroAudioSamples}, 0.0f);

    const auto start = std::chrono::high_resolution_clock::now();
    timeSection("Predictor", [&]() {
      _mlir_ciface_forward_predictor(&predictorRet, &params0, &rngState,
                                     &inputIds, &refStyle);
    });
    const size_t frameCount = buildFrameIndices(predictorRet, frameIndices);
    printLogLabel();
    std::cout << "Built duration alignment with " << frameCount
              << " fixed frames" << std::endl;
    timeSection("Vocoder", [&]() {
      _mlir_ciface_forward_vocoder(&audio, &params1, &inputIds, &frameIndices,
                                   &predictorRet.durationHidden,
                                   &predictorRet.style, &predictorRet.textMask,
                                   &refStyle);
    });
    printAudioStats(audio);
    const auto end = std::chrono::high_resolution_clock::now();

    timeSection("Save audio", [&]() { saveWav(outputPath, std::move(audio)); });

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << std::endl;
    printLogLabel();
    std::cout << "Kokoro static inference finished in "
              << elapsed.count() / 1000.0 << "s" << std::endl;
    printLogLabel();
    std::cout << "Output wav: " << std::filesystem::absolute(outputPath)
              << std::endl;
    printLogLabel();
    std::cout << "First 8 predicted durations:";
    for (size_t i = 0; i < std::min<size_t>(8, KokoroSeqLen); ++i)
      std::cout << " " << predictorRet.predDur.getData()[i];
    std::cout << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "\033[31;1m[Error] \033[0m" << e.what() << "\n";
    return 1;
  }
  return 0;
}
