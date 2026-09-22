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
//   prompt text -> tokenizeQwen3VL (pure C++) -> input_ids
//   image (pre-processed pixel_values) -> [compiled vision encoder .so]
//     -> image embeds + 3 deepstack features
//     -> splice into the tied embedding table at <image> token positions
//     -> [compiled decoder .so] greedy decode (see modes below)
//     -> revertQwen3 detokenize -> text.
//
// Decoder modes (auto-detected from the loaded decoder_shim.so):
//   KV (default build, BUDDY_QWEN3_VL_KV_DECODE=ON):
//     qwen3vl_decoder_prefill once, then qwen3vl_decoder_decode per token
//     against fixed-length GQA caches. Prefill uses decoder_weights.data;
//     decode prefers decoder_decode_weights.data (panel-packed for GEMV).
//   Legacy full forward:
//     qwen3vl_decoder recomputes the full padded sequence every step.
//
// The compiled kernels are loaded as separate shim shared libraries
// (vision_shim.so / decoder_shim.so) so their `_mlir_ciface_forward*` symbols
// do not clash across vision vs decoder. Activations, weights, and positional
// tables use the flat IEEE fp16 ABI those shims export (see F16Bits.h). All
// artifacts are read from the directory that contains the .rax (resolved from
// cfg.raxPath), mirroring how Whisper bundles audio.wav.
//
// Both the text and image sides are pure C++, with no Python runtime
// involved at all:
//   - Text tokenization/detokenization: Text<size_t,2>::tokenizeQwen3VL /
//     revertQwen3 in buddy/LLM/TextContainer.h.
//   - Image preprocessing: qwen3vl_image::preprocessImage in
//     ImagePreprocess.h (stb_image decode + a Pillow-bicubic-matching
//     resize + the Qwen2VL patch/merge reshape).
// For the pinned [1,14,28] vision grid, img_pos and the MRoPE
// cos/sin/causal-mask tables are query-independent model constants bundled
// at stage time (see cmd_preprocess in codegen/qwen3_vl_codegen.py); only
// pixel_values genuinely varies per query, and it is now computed in-memory
// instead of shelling out to preprocess.sh.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/Qwen3VLRunner.h"
#include "F16Bits.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"

#define STB_IMAGE_IMPLEMENTATION
#include "ImagePreprocess.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

using F16 = uint16_t;
using VisionFn = void (*)(const F16 *, long, const F16 *, F16 *, F16 *, F16 *,
                          F16 *);
using DecoderFn = void (*)(const F16 *, long, const F16 *, const F16 *,
                           const F16 *, const F16 *, const F16 *, const F16 *,
                           const F16 *, F16 *, long, long, long, long);
// Matches decoder_kv_shim.cpp: prefill fills 56 GQA caches + full logits.
using DecoderPrefillFn = void (*)(const F16 *, long, const F16 *, const F16 *,
                                  const F16 *, const F16 *, const F16 *,
                                  const F16 *, const F16 *, F16 *, F16 *, long,
                                  long, long, long, long);
// One-token decode: may use packed weights; updates caches in place.
using DecoderDecodeFn = void (*)(const F16 *, long, const F16 *, const F16 *,
                                 const F16 *, const F16 *, const F16 *,
                                 const F16 *, const F16 *, long, F16 *, F16 *,
                                 long, long, long, long, long);

constexpr F16 kF16NegInf = 0xfc00u;

/// Causal mask for decode step at absolute index `pos`: length-N row with
/// 0 for visible keys [0, pos] and -inf elsewhere (shape [1,1,1,N] in shim).
void fillDecodeCmask(std::vector<F16> &mask, size_t N, size_t pos) {
  mask.assign(N, kF16NegInf);
  const F16 zero = f32ToF16Bits(0.f);
  const size_t visible = pos + 1;
  if (visible > N)
    throw std::runtime_error("decode position exceeds max sequence length");
  for (size_t j = 0; j < visible; ++j)
    mask[j] = zero;
}

// mmap a file of raw IEEE fp16 bits; count is the number of elements.
struct MappedF16 {
  const F16 *data = nullptr;
  size_t count = 0;
  void *base = nullptr;
  size_t bytes = 0;
  ~MappedF16() {
    if (base && base != MAP_FAILED)
      munmap(base, bytes);
  }
};

void mapF16(const std::string &path, MappedF16 &out) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("cannot open " + path);
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error("fstat failed " + path);
  }
  out.bytes = st.st_size;
  if (out.bytes == 0 || out.bytes % sizeof(F16) != 0) {
    close(fd);
    throw std::runtime_error("invalid fp16 file size: " + path);
  }
  out.base = mmap(nullptr, out.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (out.base == MAP_FAILED) {
    out.base = nullptr;
    throw std::runtime_error("mmap failed " + path);
  }
  out.data = reinterpret_cast<const F16 *>(out.base);
  out.count = out.bytes / sizeof(F16);
}

std::vector<F16> readF16(const std::string &path, size_t expected = 0) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  size_t bytes = static_cast<size_t>(f.tellg());
  if (bytes % sizeof(F16) != 0)
    throw std::runtime_error("invalid fp16 file size: " + path);
  size_t n = bytes / sizeof(F16);
  if (expected && n != expected)
    throw std::runtime_error("unexpected fp16 count in " + path);
  f.seekg(0);
  std::vector<F16> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(F16));
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
  std::unordered_map<std::string, std::string> constants;
  std::unordered_map<std::string, std::string> codeObjects;
  std::string vocabPath;

  // All remaining resources (pixel_values is now produced in-memory by the
  // pure-C++ preprocessor, and input_ids by the pure-C++ tokenizer) are
  // query-independent bundled constants.
  std::string file(const char *name, const char *fallback) const {
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
    // All bundled resources (including embed-payload extraction) land in the
    // same directory; anchor workDir off any one resolved path.
    if (!manifest.soPath.empty())
      pkg.workDir = fs::absolute(fs::path(manifest.soPath)).parent_path();
  }
  const bool quiet = cfg.suppressStats;
  auto log = [&](const std::string &s) {
    if (!quiet)
      std::cerr << "\033[34;1m[qwen3_vl]\033[0m " << s << "\n";
  };

  std::string prompt =
      cfg.prompt.empty() ? "Read all the text in the image." : cfg.prompt;

  using Clock = std::chrono::high_resolution_clock;
  const auto e2eStart = Clock::now();
  double preprocessMs = 0.0;
  double visionMs = 0.0;
  double mergeMs = 0.0;
  double decoderTotalMs = 0.0;

  // ── Image preprocessing, pure C++. For a real --image, decode + resize +
  // patchify it directly (see ImagePreprocess.h). With no --image (or
  // --image bundled) fall back to the pixel_values.bin baked in at stage
  // time for the packaged test image. Either way, no Python is invoked. ──
  const bool hasCustomImage =
      !cfg.imagePath.empty() && cfg.imagePath != "bundled";
  std::vector<F16> pixel;
  if (hasCustomImage) {
    log("preprocessing image (pure C++): " + cfg.imagePath);
    const auto t0 = Clock::now();
    std::vector<float> pixelF32;
    qwen3vl_image::preprocessImage(cfg.imagePath, pixelF32);
    pixel = f32ToF16Bits(pixelF32);
    preprocessMs =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }

  // ── Dimensions (model constants, bundled and query-independent). ──
  size_t N = 160, NIMG = 98, HID = 2048, VOCAB = 151936;
  const size_t HEAD_DIM = 128;
  {
    std::ifstream m(pkg.file("meta", "meta.txt"));
    size_t staleS0 = 0; // legacy first field, no longer used.
    if (m)
      m >> staleS0 >> N >> NIMG >> HID >> VOCAB;
    if (!m)
      throw std::runtime_error("cannot read qwen3_vl meta.txt");
  }
  if (N == 0 || NIMG == 0 || HID == 0 || VOCAB == 0)
    throw std::runtime_error("invalid qwen3_vl dimensions in meta.txt");
  const std::vector<int> EOS = {151645, 151643};

  // ── Pure-C++ tokenization of the chat-template prompt. No Python or
  // network access is needed for this step, so a standalone .rax can serve
  // arbitrary prompts against the bundled image without a Python runtime. ──
  Text<size_t, 2> promptTok(prompt);
  promptTok.tokenizeQwen3VL(pkg.vocab(), N, NIMG);
  const size_t S0 = promptTok.getTokenCnt();
  if (S0 == 0 || S0 > N)
    throw std::runtime_error(
        "qwen3_vl tokenizer produced an invalid sequence length");

  log("loading compiled kernels + weights from " + pkg.workDir.string());
  void *visLib = dlopenOrThrow(pkg.code("vision_kernels", "vision_shim.so"));
  // Optional override for side-by-side A/B of KV vs legacy decoder .so.
  const char *decSoEnv = std::getenv("QWEN3_VL_DECODER_SO");
  const std::string decSoPath =
      decSoEnv && decSoEnv[0] ? decSoEnv
                              : pkg.code("decoder_kernels", "decoder_shim.so");
  void *decLib = dlopenOrThrow(decSoPath);
  auto visionRun = reinterpret_cast<VisionFn>(dlsym(visLib, "qwen3vl_vision"));
  auto decoderRun =
      reinterpret_cast<DecoderFn>(dlsym(decLib, "qwen3vl_decoder"));
  auto decoderPrefill = reinterpret_cast<DecoderPrefillFn>(
      dlsym(decLib, "qwen3vl_decoder_prefill"));
  auto decoderDecode = reinterpret_cast<DecoderDecodeFn>(
      dlsym(decLib, "qwen3vl_decoder_decode"));
  // Prefer KV when both entry points exist (default CMake package).
  const bool kvDecode = decoderPrefill && decoderDecode;
  if (!visionRun || (!decoderRun && !kvDecode))
    throw std::runtime_error("missing shim entry points");
  if (kvDecode)
    log("using KV prefill/decode decoder shim");

  MappedF16 Wv, Wd, WdDecode, embed;
  mapF16(pkg.file("vision_weights", "vision_weights.data"), Wv);
  // decoder_weights.data is prefill (or legacy full-forward) row-major blob.
  mapF16(pkg.file("decoder_weights", "decoder_weights.data"), Wd);
  mapF16(pkg.file("embed_table", "embed_table.bin"), embed);
  const F16 *decodeWeights = Wd.data;
  long decodeWeightCount = (long)Wd.count;
  if (kvDecode) {
    // Packed decode GEMV weights: env override, else rax/workDir resource.
    std::string decodeWeightPath;
    if (const char *env = std::getenv("QWEN3_VL_DECODE_WEIGHTS")) {
      if (env[0])
        decodeWeightPath = env;
    }
    if (decodeWeightPath.empty()) {
      const std::string candidate =
          pkg.file("decoder_decode_weights", "decoder_decode_weights.data");
      if (fs::exists(candidate))
        decodeWeightPath = candidate;
    }
    if (!decodeWeightPath.empty()) {
      mapF16(decodeWeightPath, WdDecode);
      decodeWeights = WdDecode.data;
      decodeWeightCount = (long)WdDecode.count;
      log("using packed decode weights from " + decodeWeightPath);
    }
  }
  if (hasCustomImage) {
    if (pixel.size() != 392 * 1536)
      throw std::runtime_error("qwen3_vl image preprocessing produced an "
                               "unexpected pixel_values size");
  } else {
    const auto t0 = Clock::now();
    pixel = readF16(pkg.file("pixel_values", "pixel_values.bin"), 392 * 1536);
    preprocessMs =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }
  std::vector<int64_t> inputIds(S0);
  for (size_t i = 0; i < S0; ++i)
    inputIds[i] = static_cast<int64_t>(promptTok[i]);
  std::vector<int64_t> imgPos =
      readI64(pkg.file("img_pos", "img_pos.i64"), NIMG);
  std::vector<F16> cosv =
      readF16(pkg.file("cos", "cos.bin"), checkedMul(N, HEAD_DIM, "cos"));
  std::vector<F16> sinv =
      readF16(pkg.file("sin", "sin.bin"), checkedMul(N, HEAD_DIM, "sin"));
  std::vector<F16> cmask =
      readF16(pkg.file("cmask", "cmask.bin"),
              checkedMul(checkedMul(N, N, "cmask"), 1, "cmask"));
  if (embed.count < checkedMul(VOCAB, HID, "embed_table"))
    throw std::runtime_error("embed_table.bin is smaller than meta dimensions");
  for (int64_t tok : inputIds)
    if (tok < 0 || static_cast<size_t>(tok) >= VOCAB)
      throw std::runtime_error("tokenized prompt contains token out of range");
  for (int64_t pos : imgPos)
    if (pos < 0 || static_cast<size_t>(pos) >= N)
      throw std::runtime_error("img_pos.i64 contains position out of range");
  auto embedRow = [&](int64_t tok) { return embed.data + (size_t)tok * HID; };

  // ── 1. Compiled vision encoder. ──
  log("running compiled vision encoder ...");
  std::vector<F16> pooled(NIMG * HID);
  std::vector<F16> ds0(NIMG * HID), ds1(NIMG * HID), ds2(NIMG * HID);
  {
    const auto t0 = Clock::now();
    visionRun(Wv.data, (long)Wv.count, pixel.data(), ds0.data(), ds1.data(),
              ds2.data(), pooled.data());
    visionMs =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }

  // ── 2. Build inputs_embeds[N,HID] and pre-scattered deepstack-full. ──
  std::vector<F16> ie(N * HID, 0);
  std::vector<F16> df0(N * HID, 0), df1(N * HID, 0), df2(N * HID, 0);
  {
    const auto t0 = Clock::now();
    for (size_t i = 0; i < S0; ++i)
      std::copy(embedRow(inputIds[i]), embedRow(inputIds[i]) + HID,
                ie.begin() + i * HID);
    for (size_t k = 0; k < imgPos.size(); ++k)
      std::copy(pooled.begin() + k * HID, pooled.begin() + (k + 1) * HID,
                ie.begin() + (size_t)imgPos[k] * HID);
    for (size_t k = 0; k < imgPos.size(); ++k) {
      size_t off = (size_t)imgPos[k] * HID;
      std::copy(ds0.begin() + k * HID, ds0.begin() + (k + 1) * HID,
                df0.begin() + off);
      std::copy(ds1.begin() + k * HID, ds1.begin() + (k + 1) * HID,
                df1.begin() + off);
      std::copy(ds2.begin() + k * HID, ds2.begin() + (k + 1) * HID,
                df2.begin() + off);
    }
    mergeMs =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }

  // ── 3. Greedy decode loop through the compiled decoder. ──
  log("greedy decoding through compiled decoder ...");
  Text<size_t, 2> out;
  out.loadVocab(pkg.vocab());
  std::vector<F16> logits(N * VOCAB);
  std::vector<F16> logitsStep(VOCAB);
  std::vector<F16> ieStep(HID);
  std::vector<F16> decCmask;
  std::vector<F16> df0z(HID, f32ToF16Bits(0.f));
  std::vector<F16> df1z(HID, f32ToF16Bits(0.f));
  std::vector<F16> df2z(HID, f32ToF16Bits(0.f));
  std::vector<F16> kv;
  const size_t NKV = 8;
  std::string streamedText;
  size_t generated = 0;
  const size_t maxNewTokens =
      cfg.maxNewTokens > 0 ? static_cast<size_t>(cfg.maxNewTokens) : N;
  const auto decoderStart = Clock::now();
  auto emitGenerated = [&](int tok, double stepSecs) {
    bool isEos = false;
    for (int e : EOS)
      isEos |= (tok == e);
    if (isEos)
      return false;
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
    return true;
  };

  if (kvDecode) {
    // 56 = 28 layers × {K,V}; each block is [1, NKV, N, HEAD_DIM] f16.
    kv.resize(checkedMul(checkedMul(checkedMul(56, NKV, "kv"), N, "kv"),
                         HEAD_DIM, "kv"));
    const auto prefillStart = Clock::now();
    // Prefill always uses the row-major prefill weight blob (Wd).
    decoderPrefill(Wd.data, (long)Wd.count, ie.data(), cosv.data(), sinv.data(),
                   cmask.data(), df0.data(), df1.data(), df2.data(), kv.data(),
                   logits.data(), (long)N, (long)VOCAB, (long)HID,
                   (long)HEAD_DIM, (long)NKV);
    const double prefillSecs =
        std::chrono::duration<double>(Clock::now() - prefillStart).count();
    if (!quiet)
      std::cout << "[KV prefill] " << prefillSecs << "s\n";

    // First generated token = argmax at the last prompt position.
    const size_t t0 = S0 - 1;
    int tok = argmaxF16(logits.data() + t0 * VOCAB, VOCAB);
    if (emitGenerated(tok, prefillSecs)) {
      while (generated < maxNewTokens) {
        // Absolute index of the new token we are about to write into the cache.
        const size_t pos = S0 + generated - 1;
        if (pos >= N)
          break;
        const auto stepStart = Clock::now();
        std::copy(embedRow(tok), embedRow(tok) + HID, ieStep.data());
        fillDecodeCmask(decCmask, N, pos);
        // decodeWeights may be the packed panel layout; deepstack zeros for
        // steps after prefill (image fusion already applied).
        decoderDecode(decodeWeights, decodeWeightCount, ieStep.data(),
                      cosv.data() + pos * HEAD_DIM,
                      sinv.data() + pos * HEAD_DIM, decCmask.data(),
                      df0z.data(), df1z.data(), df2z.data(), (long)pos,
                      kv.data(), logitsStep.data(), (long)N, (long)VOCAB,
                      (long)HID, (long)HEAD_DIM, (long)NKV);
        const double stepSecs =
            std::chrono::duration<double>(Clock::now() - stepStart).count();
        tok = argmaxF16(logitsStep.data(), VOCAB);
        if (!emitGenerated(tok, stepSecs))
          break;
      }
    }
  } else {
    // Legacy: full-sequence recompute every greedy step (no KV).
    for (size_t t = S0 - 1; t + 1 < N && generated < maxNewTokens; ++t) {
      const auto stepStart = Clock::now();
      decoderRun(Wd.data, (long)Wd.count, ie.data(), cosv.data(), sinv.data(),
                 cmask.data(), df0.data(), df1.data(), df2.data(),
                 logits.data(), (long)N, (long)VOCAB, (long)HID,
                 (long)HEAD_DIM);
      const double stepSecs =
          std::chrono::duration<double>(Clock::now() - stepStart).count();
      const F16 *row = logits.data() + t * VOCAB;
      int tok = argmaxF16(row, VOCAB);
      if (!emitGenerated(tok, stepSecs))
        break;
      std::copy(embedRow(tok), embedRow(tok) + HID, ie.begin() + (t + 1) * HID);
    }
  }
  decoderTotalMs =
      std::chrono::duration<double, std::milli>(Clock::now() - decoderStart)
          .count();

  // ── 4. Detokenize + print. ──
  std::string text = out.revertQwen3();
  std::cout << "\033[33;1m[Qwen3-VL OCR]\033[0m " << text << std::endl;

  const double e2eMs =
      std::chrono::duration<double, std::milli>(Clock::now() - e2eStart)
          .count();
  const double tokensPerSec =
      generated > 0 && decoderTotalMs > 0.0
          ? (1000.0 * static_cast<double>(generated) / decoderTotalMs)
          : 0.0;
  std::cerr << "[stage] precision=f16\n"
            << "[stage] decoder_mode=" << (kvDecode ? "kv" : "full") << "\n"
            << "[stage] preprocess_ms=" << preprocessMs << "\n"
            << "[stage] vision_encoder_ms=" << visionMs << "\n"
            << "[stage] multimodal_merge_ms=" << mergeMs << "\n"
            << "[stage] decoder_total_ms=" << decoderTotalMs << "\n"
            << "[stage] decoder_steps=" << generated << "\n"
            << "[stage] generated_tokens=" << generated << "\n"
            << "[stage] tokens_per_sec=" << tokensPerSec << "\n"
            << "[stage] e2e_ms=" << e2eMs << "\n";

  dlclose(visLib);
  dlclose(decLib);
}

} // namespace runtime
} // namespace buddy
