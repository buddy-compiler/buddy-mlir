//===- Qwen3VLRunner.cpp - Qwen3-VL end-to-end inference loop -------------===//
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
// Drives the full Qwen3-VL OCR pipeline through the buddy-compiled kernels:
//   image (pre-processed pixel_values) -> [compiled vision encoder .so]
//     -> image embeds + 3 deepstack features
//     -> splice into the tied embedding table at <image> token positions
//     -> [compiled decoder .so] greedy decode loop (fixed MAX seq, no KV cache)
//     -> revertQwen3 detokenize -> text.
//
// The compiled kernels are loaded as separate shim shared libraries
// (vision_shim.so / decoder_shim.so) so their identical `_mlir_ciface_forward`
// symbols do not clash. All artifacts are read from the directory that contains
// the .rax (resolved from cfg.raxPath), mirroring how Whisper bundles
// audio.wav.
//
// Per-query image preprocessing is delegated to the packaged Python helper so
// the compiled model still receives the fixed-grid tensors it was imported for.
// A pure-C++ preprocessing path is future work.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/Qwen3VLRunner.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

using VisionFn = void (*)(const float *, long, const float *, float *, float *,
                          float *, float *);
using DecoderFn = void (*)(const float *, long, const float *, const float *,
                           const float *, const float *, const float *,
                           const float *, const float *, float *, long, long,
                           long, long);

// mmap a file as read-only float32; returns base pointer and element count.
struct MappedFloats {
  const float *data = nullptr;
  size_t count = 0;
  void *base = nullptr;
  size_t bytes = 0;
  ~MappedFloats() {
    if (base && base != MAP_FAILED)
      munmap(base, bytes);
  }
};

void mapFloats(const std::string &path, MappedFloats &out) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("cannot open " + path);
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error("fstat failed " + path);
  }
  out.bytes = st.st_size;
  if (out.bytes == 0 || out.bytes % sizeof(float) != 0) {
    close(fd);
    throw std::runtime_error("invalid float file size: " + path);
  }
  out.base = mmap(nullptr, out.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (out.base == MAP_FAILED) {
    out.base = nullptr;
    throw std::runtime_error("mmap failed " + path);
  }
  out.data = reinterpret_cast<const float *>(out.base);
  out.count = out.bytes / sizeof(float);
}

std::vector<float> readFloats(const std::string &path, size_t expected = 0) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  size_t bytes = static_cast<size_t>(f.tellg());
  if (bytes % sizeof(float) != 0)
    throw std::runtime_error("invalid float file size: " + path);
  size_t n = bytes / sizeof(float);
  if (expected && n != expected)
    throw std::runtime_error("unexpected float count in " + path);
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  if (!f)
    throw std::runtime_error("short read from " + path);
  return v;
}

std::vector<int64_t> readI64(const std::string &path, size_t expected = 0) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  size_t bytes = static_cast<size_t>(f.tellg());
  if (bytes % sizeof(int64_t) != 0)
    throw std::runtime_error("invalid int64 file size: " + path);
  size_t n = bytes / sizeof(int64_t);
  if (expected && n != expected)
    throw std::runtime_error("unexpected int64 count in " + path);
  f.seekg(0);
  std::vector<int64_t> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(int64_t));
  if (!f)
    throw std::runtime_error("short read from " + path);
  return v;
}

void *dlopenOrThrow(const std::string &path) {
  void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h)
    throw std::runtime_error("dlopen " + path + ": " + dlerror());
  return h;
}

struct Qwen3VLPackage {
  fs::path workDir;
  fs::path queryDir;
  std::unordered_map<std::string, std::string> constants;
  std::unordered_map<std::string, std::string> codeObjects;
  std::string vocabPath;

  std::string file(const char *name, const char *fallback) const {
    static const std::vector<std::string> dynamicNames = {
        "pixel_values", "input_ids", "img_pos", "cos", "sin", "cmask", "meta"};
    if (!queryDir.empty() && std::find(dynamicNames.begin(), dynamicNames.end(),
                                       name) != dynamicNames.end())
      return (queryDir / fallback).string();
    auto it = constants.find(name);
    if (it != constants.end())
      return it->second;
    return (workDir / fallback).string();
  }

  std::string code(const char *name, const char *fallback) const {
    auto it = codeObjects.find(name);
    if (it != codeObjects.end())
      return it->second;
    return (workDir / fallback).string();
  }

  std::string vocab() const {
    return vocabPath.empty() ? (workDir / "vocab.txt").string() : vocabPath;
  }
};

struct ScopedDir {
  fs::path path;
  ~ScopedDir() {
    if (!path.empty()) {
      std::error_code ec;
      fs::remove_all(path, ec);
    }
  }
};

fs::path makeTempDir() {
  fs::path base = fs::temp_directory_path() / "buddy-qwen3-vl";
  fs::create_directories(base);
  for (int i = 0; i < 100; ++i) {
    fs::path candidate =
        base / (std::to_string(getpid()) + "-" + std::to_string(i));
    std::error_code ec;
    if (fs::create_directory(candidate, ec))
      return candidate;
  }
  throw std::runtime_error("failed to create qwen3_vl temporary directory");
}

void runProcess(const std::vector<std::string> &args) {
  if (args.empty())
    throw std::runtime_error("runProcess called with no arguments");

  std::vector<char *> argv;
  argv.reserve(args.size() + 1);
  for (const auto &arg : args)
    argv.push_back(const_cast<char *>(arg.c_str()));
  argv.push_back(nullptr);

  pid_t pid = fork();
  if (pid < 0)
    throw std::runtime_error(std::string("fork failed: ") + strerror(errno));
  if (pid == 0) {
    execvp(argv[0], argv.data());
    std::_Exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0)
    throw std::runtime_error(std::string("waitpid failed: ") + strerror(errno));
  if (!WIFEXITED(status))
    throw std::runtime_error("preprocess.sh did not exit normally");
  if (WEXITSTATUS(status) != 0)
    throw std::runtime_error("preprocess.sh failed with exit code " +
                             std::to_string(WEXITSTATUS(status)));
}

size_t checkedMul(size_t a, size_t b, const char *what) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    throw std::runtime_error(std::string("size overflow: ") + what);
  return a * b;
}

std::vector<std::string> splitUtf8Chars(const std::string &text) {
  std::vector<std::string> chars;
  for (size_t i = 0; i < text.size();) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    size_t len = 1;
    if ((c & 0xE0) == 0xC0)
      len = 2;
    else if ((c & 0xF0) == 0xE0)
      len = 3;
    else if ((c & 0xF8) == 0xF0)
      len = 4;
    if (i + len > text.size())
      len = 1;
    chars.emplace_back(text.substr(i, len));
    i += len;
  }
  return chars;
}

std::string displayChar(const std::string &text) {
  if (text == "\n")
    return "\\n";
  if (text == "\r")
    return "\\r";
  if (text == "\t")
    return "\\t";
  return text;
}

} // namespace

namespace buddy {
namespace runtime {

void Qwen3VLRunner::run(const RunConfig &cfg) {
  // ── Resolve the package directory (where the .rax + artifacts live). ──
  Qwen3VLPackage pkg;
  pkg.workDir = cfg.raxPath.empty()
                    ? fs::current_path()
                    : fs::absolute(fs::path(cfg.raxPath)).parent_path();
  if (!cfg.raxPath.empty()) {
    auto manifest = ModelManifest::loadFromRax(cfg.raxPath);
    for (const auto &constant : manifest.constants)
      pkg.constants[constant.name] = constant.path;
    for (const auto &codeObject : manifest.codeObjects)
      pkg.codeObjects[codeObject.name] = codeObject.path;
    pkg.vocabPath = manifest.vocabPath;
    auto it = pkg.constants.find("preprocess_script");
    if (it != pkg.constants.end())
      pkg.workDir = fs::absolute(fs::path(it->second)).parent_path();
  }
  const bool quiet = cfg.suppressStats;
  auto log = [&](const std::string &s) {
    if (!quiet)
      std::cerr << "\033[34;1m[qwen3_vl]\033[0m " << s << "\n";
  };

  // ── Per-query preprocessing for a real --image (HF processor via Python). ──
  // Produces pixel_values/input_ids/img_pos/cos/sin/cmask/meta in `dir`. With
  // no --image (or --image bundled) the runner uses whatever is already staged.
  ScopedDir query;
  if (!cfg.imagePath.empty() && cfg.imagePath != "bundled") {
    std::string prompt =
        cfg.prompt.empty() ? "Read all the text in the image." : cfg.prompt;
    query.path = makeTempDir();
    pkg.queryDir = query.path;
    log("preprocessing image: " + cfg.imagePath);
    runProcess({"bash", pkg.file("preprocess_script", "preprocess.sh"),
                cfg.imagePath, prompt, pkg.queryDir.string()});
  }

  // ── Dimensions (per-query seq from meta; rest are model constants). ──
  size_t S0 = 116, N = 160, NIMG = 98, HID = 2048, VOCAB = 151936;
  const size_t HEAD_DIM = 128;
  {
    std::ifstream m(pkg.file("meta", "meta.txt"));
    if (m)
      m >> S0 >> N >> NIMG >> HID >> VOCAB;
    if (!m)
      throw std::runtime_error("cannot read qwen3_vl meta.txt");
  }
  if (S0 == 0 || S0 > N || N == 0 || NIMG == 0 || HID == 0 || VOCAB == 0)
    throw std::runtime_error("invalid qwen3_vl dimensions in meta.txt");
  const std::vector<int> EOS = {151645, 151643};

  log("loading compiled kernels + weights from " + pkg.workDir.string());
  void *visLib = dlopenOrThrow(pkg.code("vision_kernels", "vision_shim.so"));
  void *decLib = dlopenOrThrow(pkg.code("decoder_kernels", "decoder_shim.so"));
  auto visionRun = reinterpret_cast<VisionFn>(dlsym(visLib, "qwen3vl_vision"));
  auto decoderRun =
      reinterpret_cast<DecoderFn>(dlsym(decLib, "qwen3vl_decoder"));
  if (!visionRun || !decoderRun)
    throw std::runtime_error("missing shim entry points");

  MappedFloats Wv, Wd, embed;
  mapFloats(pkg.file("vision_weights", "vision_weights.data"), Wv);
  mapFloats(pkg.file("decoder_weights", "decoder_weights.data"), Wd);
  mapFloats(pkg.file("embed_table", "embed_table.bin"), embed);
  std::vector<float> pixel =
      readFloats(pkg.file("pixel_values", "pixel_values.bin"), 392 * 1536);
  std::vector<int64_t> inputIds =
      readI64(pkg.file("input_ids", "input_ids.i64"), S0);
  std::vector<int64_t> imgPos =
      readI64(pkg.file("img_pos", "img_pos.i64"), NIMG);
  std::vector<float> cosv =
      readFloats(pkg.file("cos", "cos.bin"), checkedMul(N, HEAD_DIM, "cos"));
  std::vector<float> sinv =
      readFloats(pkg.file("sin", "sin.bin"), checkedMul(N, HEAD_DIM, "sin"));
  std::vector<float> cmask =
      readFloats(pkg.file("cmask", "cmask.bin"),
                 checkedMul(checkedMul(N, N, "cmask"), 1, "cmask"));
  if (embed.count < checkedMul(VOCAB, HID, "embed_table"))
    throw std::runtime_error("embed_table.bin is smaller than meta dimensions");
  for (int64_t tok : inputIds)
    if (tok < 0 || static_cast<size_t>(tok) >= VOCAB)
      throw std::runtime_error("input_ids.i64 contains token out of range");
  for (int64_t pos : imgPos)
    if (pos < 0 || static_cast<size_t>(pos) >= N)
      throw std::runtime_error("img_pos.i64 contains position out of range");
  auto embedRow = [&](int64_t tok) { return embed.data + (size_t)tok * HID; };

  // ── 1. Compiled vision encoder. ──
  log("running compiled vision encoder ...");
  std::vector<float> pooled(NIMG * HID);
  std::vector<float> ds0(NIMG * HID), ds1(NIMG * HID), ds2(NIMG * HID);
  visionRun(Wv.data, (long)Wv.count, pixel.data(), ds0.data(), ds1.data(),
            ds2.data(), pooled.data());

  // ── 2. Build inputs_embeds[N,HID] and pre-scattered deepstack-full. ──
  std::vector<float> ie(N * HID, 0.f);
  for (size_t i = 0; i < S0; ++i)
    std::copy(embedRow(inputIds[i]), embedRow(inputIds[i]) + HID,
              ie.begin() + i * HID);
  for (size_t k = 0; k < imgPos.size(); ++k)
    std::copy(pooled.begin() + k * HID, pooled.begin() + (k + 1) * HID,
              ie.begin() + (size_t)imgPos[k] * HID);
  std::vector<float> df0(N * HID, 0.f), df1(N * HID, 0.f), df2(N * HID, 0.f);
  for (size_t k = 0; k < imgPos.size(); ++k) {
    size_t off = (size_t)imgPos[k] * HID;
    std::copy(ds0.begin() + k * HID, ds0.begin() + (k + 1) * HID,
              df0.begin() + off);
    std::copy(ds1.begin() + k * HID, ds1.begin() + (k + 1) * HID,
              df1.begin() + off);
    std::copy(ds2.begin() + k * HID, ds2.begin() + (k + 1) * HID,
              df2.begin() + off);
  }

  // ── 3. Greedy decode loop through the compiled decoder. ──
  log("greedy decoding through compiled decoder ...");
  Text<size_t, 2> out;
  out.loadVocab(pkg.vocab());
  std::vector<float> logits(N * VOCAB);
  std::string streamedText;
  size_t generated = 0;
  const size_t maxNewTokens =
      cfg.maxNewTokens > 0 ? static_cast<size_t>(cfg.maxNewTokens) : N;
  for (size_t t = S0 - 1; t + 1 < N && generated < maxNewTokens; ++t) {
    const auto stepStart = std::chrono::high_resolution_clock::now();
    decoderRun(Wd.data, (long)Wd.count, ie.data(), cosv.data(), sinv.data(),
               cmask.data(), df0.data(), df1.data(), df2.data(), logits.data(),
               (long)N, (long)VOCAB, (long)HID, (long)HEAD_DIM);
    const double stepSecs =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stepStart)
            .count();
    const float *row = logits.data() + t * VOCAB;
    int tok = 0;
    for (size_t j = 1; j < VOCAB; ++j)
      if (row[j] > row[tok])
        tok = (int)j;
    bool isEos = false;
    for (int e : EOS)
      isEos |= (tok == e);
    if (isEos)
      break;
    if (tok < 0 || static_cast<size_t>(tok) >= VOCAB)
      throw std::runtime_error("decoder produced token out of range");
    out.appendTokenIdx((size_t)tok);
    ++generated;
    const std::string currentText = out.revertQwen3();
    std::string delta;
    if (currentText.size() > streamedText.size())
      delta.assign(currentText.data() + streamedText.size(),
                   currentText.size() - streamedText.size());
    streamedText = currentText;
    if (!quiet) {
      for (const std::string &ch : splitUtf8Chars(delta)) {
        std::cout << "\033[32;1m[Iteration " << generated << "]\033[0m "
                  << "Char: " << displayChar(ch) << " | Token: " << tok
                  << " | Time: " << stepSecs << "s\n";
      }
      std::cout.flush();
    }
    std::copy(embedRow(tok), embedRow(tok) + HID, ie.begin() + (t + 1) * HID);
  }

  // ── 4. Detokenize + print. ──
  std::string text = out.revertQwen3();
  std::cout << "\033[33;1m[Qwen3-VL OCR]\033[0m " << text << std::endl;

  dlclose(visLib);
  dlclose(decLib);
}

} // namespace runtime
} // namespace buddy
