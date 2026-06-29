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
// NOTE: on-the-fly image preprocessing (PNG decode + Qwen smart-resize +
// patchify) is not yet implemented; the runner consumes a bundled, pre-computed
// pixel_values.bin. Everything downstream is the real compiled model.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/Qwen3VLRunner.h"
#include "buddy/LLM/TextContainer.h"

#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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
    if (base)
      munmap(base, bytes);
  }
};

void mapFloats(const std::string &path, MappedFloats &out) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("cannot open " + path);
  struct stat st;
  fstat(fd, &st);
  out.bytes = st.st_size;
  out.base = mmap(nullptr, out.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (out.base == MAP_FAILED)
    throw std::runtime_error("mmap failed " + path);
  out.data = reinterpret_cast<const float *>(out.base);
  out.count = out.bytes / sizeof(float);
}

std::vector<float> readFloats(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  size_t n = f.tellg() / sizeof(float);
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  return v;
}

std::vector<int64_t> readI64(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  size_t n = f.tellg() / sizeof(int64_t);
  f.seekg(0);
  std::vector<int64_t> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(int64_t));
  return v;
}

void *dlopenOrThrow(const std::string &path) {
  void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h)
    throw std::runtime_error("dlopen " + path + ": " + dlerror());
  return h;
}

} // namespace

namespace buddy {
namespace runtime {

void Qwen3VLRunner::run(const RunConfig &cfg) {
  // ── Resolve the package directory (where the .rax + artifacts live). ──
  fs::path dir = cfg.raxPath.empty()
                     ? fs::current_path()
                     : fs::absolute(fs::path(cfg.raxPath)).parent_path();
  auto P = [&](const char *n) { return (dir / n).string(); };
  const bool quiet = cfg.suppressStats;
  auto log = [&](const std::string &s) {
    if (!quiet)
      std::cerr << "\033[34;1m[qwen3_vl]\033[0m " << s << "\n";
  };

  // ── Per-query preprocessing for a real --image (HF processor via Python). ──
  // Produces pixel_values/input_ids/img_pos/cos/sin/cmask/meta in `dir`. With
  // no --image (or --image bundled) the runner uses whatever is already staged.
  if (!cfg.imagePath.empty() && cfg.imagePath != "bundled") {
    std::string prompt =
        cfg.prompt.empty() ? "Read all the text in the image." : cfg.prompt;
    std::string cmd = "bash '" + P("preprocess.sh") + "' '" + cfg.imagePath +
                      "' '" + prompt + "' '" + dir.string() + "'";
    log("preprocessing image: " + cfg.imagePath);
    if (std::system(cmd.c_str()) != 0)
      throw std::runtime_error("preprocess.sh failed");
  }

  // ── Dimensions (per-query seq from meta; rest are model constants). ──
  size_t S0 = 116, N = 160, NIMG = 98, HID = 2048, VOCAB = 151936;
  const size_t HEAD_DIM = 128;
  {
    std::ifstream m(P("meta.txt"));
    if (m)
      m >> S0 >> N >> NIMG >> HID >> VOCAB;
  }
  const std::vector<int> EOS = {151645, 151643};

  log("loading compiled kernels + weights from " + dir.string());
  void *visLib = dlopenOrThrow(P("vision_shim.so"));
  void *decLib = dlopenOrThrow(P("decoder_shim.so"));
  auto visionRun = reinterpret_cast<VisionFn>(dlsym(visLib, "qwen3vl_vision"));
  auto decoderRun =
      reinterpret_cast<DecoderFn>(dlsym(decLib, "qwen3vl_decoder"));
  if (!visionRun || !decoderRun)
    throw std::runtime_error("missing shim entry points");

  MappedFloats Wv, Wd, embed;
  mapFloats(P("vision_weights.data"), Wv);
  mapFloats(P("decoder_weights.data"), Wd);
  mapFloats(P("embed_table.bin"), embed); // (VOCAB, HID), tied lm_head
  std::vector<float> pixel = readFloats(P("pixel_values.bin"));
  std::vector<int64_t> inputIds = readI64(P("input_ids.i64"));
  std::vector<int64_t> imgPos = readI64(P("img_pos.i64"));
  std::vector<float> cosv = readFloats(P("cos.bin")); // (N, HEAD_DIM)
  std::vector<float> sinv = readFloats(P("sin.bin"));
  std::vector<float> cmask = readFloats(P("cmask.bin")); // (1,1,N,N)
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
  out.loadVocab(P("vocab.txt"));
  std::vector<float> logits(N * VOCAB);
  for (size_t t = S0 - 1; t + 1 < N; ++t) {
    decoderRun(Wd.data, (long)Wd.count, ie.data(), cosv.data(), sinv.data(),
               cmask.data(), df0.data(), df1.data(), df2.data(), logits.data(),
               (long)N, (long)VOCAB, (long)HID, (long)HEAD_DIM);
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
    out.appendTokenIdx((size_t)tok);
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
