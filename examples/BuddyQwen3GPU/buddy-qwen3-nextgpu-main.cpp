#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <sys/time.h>
#endif

extern "C" double _mlir_ciface_rtclock() {
#ifndef _WIN32
  struct timeval tp;
  int stat = gettimeofday(&tp, nullptr);
  if (stat != 0)
    return 0.0;
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
  return 0.0;
#endif
}

template <typename T, int N> struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

struct PrefillReturn {
  StridedMemRefType<float, 3> hidden;
  StridedMemRefType<float, 4> k_cache;
  StridedMemRefType<float, 4> v_cache;
};

struct DecodeReturn {
  StridedMemRefType<float, 3> hidden;
  StridedMemRefType<float, 4> k_cache;
  StridedMemRefType<float, 4> v_cache;
};

extern "C" void _mlir_ciface_qwen3_prefill_kernel(
    PrefillReturn *result, StridedMemRefType<float, 3> *x,
    StridedMemRefType<float, 3> *norm1_w, StridedMemRefType<float, 3> *wq,
    StridedMemRefType<float, 3> *wk, StridedMemRefType<float, 3> *wv,
    StridedMemRefType<float, 4> *q_norm, StridedMemRefType<float, 4> *k_norm,
    StridedMemRefType<float, 3> *wo, StridedMemRefType<float, 4> *attn_mask,
    StridedMemRefType<float, 4> *cos, StridedMemRefType<float, 4> *sin,
    StridedMemRefType<float, 3> *norm2_w, StridedMemRefType<float, 3> *ffn_gate,
    StridedMemRefType<float, 3> *ffn_up, StridedMemRefType<float, 3> *ffn_down);

extern "C" void _mlir_ciface_qwen3_decode_kernel(
    DecodeReturn *result, StridedMemRefType<float, 3> *x,
    StridedMemRefType<float, 3> *norm1_w, StridedMemRefType<float, 3> *wq,
    StridedMemRefType<float, 3> *wk, StridedMemRefType<float, 3> *wv,
    StridedMemRefType<float, 4> *q_norm, StridedMemRefType<float, 4> *k_norm,
    StridedMemRefType<float, 3> *wo, StridedMemRefType<float, 4> *attn_mask,
    StridedMemRefType<float, 4> *cos, StridedMemRefType<float, 4> *sin,
    StridedMemRefType<int64_t, 1> *cache_pos,
    StridedMemRefType<float, 4> *k_cache, StridedMemRefType<float, 4> *v_cache,
    StridedMemRefType<float, 3> *norm2_w, StridedMemRefType<float, 3> *ffn_gate,
    StridedMemRefType<float, 3> *ffn_up, StridedMemRefType<float, 3> *ffn_down);

template <typename T, int N>
StridedMemRefType<T, N> makeMemRef(T *data,
                                   const std::array<int64_t, N> &sizes) {
  StridedMemRefType<T, N> memref{};
  memref.basePtr = data;
  memref.data = data;
  memref.offset = 0;
  for (int i = 0; i < N; ++i)
    memref.sizes[i] = sizes[i];
  int64_t stride = 1;
  for (int i = N - 1; i >= 0; --i) {
    memref.strides[i] = stride;
    stride *= sizes[i];
  }
  return memref;
}

struct LayerWeights {
  float *norm1;
  float *wq;
  float *wk;
  float *wv;
  float *q_norm;
  float *k_norm;
  float *wo;
  float *norm2;
  float *ffn_gate;
  float *ffn_up;
  float *ffn_down;
};

struct Weights {
  uint32_t layers;
  uint32_t vocab;
  uint32_t hidden;
  uint32_t ffn;
  uint32_t seq;
  uint32_t heads;

  std::vector<float> storage;
  float *embed;
  float *final_norm;
  float *lm_head;
  std::vector<LayerWeights> layer;
};

static size_t checkedMul(size_t a, size_t b) {
  if (a == 0 || b <= (std::numeric_limits<size_t>::max() / a))
    return a * b;
  throw std::runtime_error("size overflow");
}

static Weights loadWeights(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("cannot open weights file: " + path);

  struct Header {
    char magic[8];
    uint32_t version;
    uint32_t layers;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t ffn;
    uint32_t seq;
    uint32_t heads;
  } h{};

  f.read(reinterpret_cast<char *>(&h), sizeof(h));
  if (!f || std::string(h.magic, h.magic + 7) != "BQ3GPUW" || h.version != 2)
    throw std::runtime_error("bad weights header");

  size_t globalCount = 0;
  globalCount += checkedMul((size_t)h.vocab, (size_t)h.hidden);
  globalCount += h.hidden;
  globalCount += checkedMul((size_t)h.hidden, (size_t)h.vocab);

  size_t perLayer = 0;
  perLayer += h.hidden;
  perLayer += checkedMul((size_t)h.hidden, (size_t)h.hidden) * 6;
  perLayer += 128 * 2;
  perLayer += h.hidden;
  perLayer += checkedMul((size_t)h.hidden, (size_t)h.ffn) * 2;
  perLayer += checkedMul((size_t)h.ffn, (size_t)h.hidden);

  size_t totalCount = globalCount + checkedMul((size_t)h.layers, perLayer);
  std::vector<float> buf(totalCount);
  f.read(reinterpret_cast<char *>(buf.data()),
         (std::streamsize)(totalCount * sizeof(float)));
  if (!f)
    throw std::runtime_error("weights file truncated");

  Weights w{};
  w.layers = h.layers;
  w.vocab = h.vocab;
  w.hidden = h.hidden;
  w.ffn = h.ffn;
  w.seq = h.seq;
  w.heads = h.heads;
  w.storage = std::move(buf);

  size_t off = 0;
  auto take = [&](size_t n) -> float * {
    float *p = w.storage.data() + off;
    off += n;
    return p;
  };

  w.embed = take(checkedMul((size_t)w.vocab, (size_t)w.hidden));
  w.final_norm = take(w.hidden);
  w.lm_head = take(checkedMul((size_t)w.hidden, (size_t)w.vocab));

  w.layer.resize(w.layers);
  for (uint32_t i = 0; i < w.layers; ++i) {
    auto &L = w.layer[i];
    L.norm1 = take(w.hidden);
    L.wq = take(checkedMul((size_t)w.hidden, (size_t)w.hidden) * 2);
    L.wk = take(checkedMul((size_t)w.hidden, (size_t)w.hidden));
    L.wv = take(checkedMul((size_t)w.hidden, (size_t)w.hidden));
    L.q_norm = take(128);
    L.k_norm = take(128);
    L.wo = take(checkedMul((size_t)w.hidden, (size_t)w.hidden) * 2);
    L.norm2 = take(w.hidden);
    L.ffn_gate = take(checkedMul((size_t)w.hidden, (size_t)w.ffn));
    L.ffn_up = take(checkedMul((size_t)w.hidden, (size_t)w.ffn));
    L.ffn_down = take(checkedMul((size_t)w.ffn, (size_t)w.hidden));
  }
  if (off != w.storage.size())
    throw std::runtime_error("weights layout mismatch");

  return w;
}

static std::vector<int64_t> parseIds(const std::string &s) {
  std::vector<int64_t> ids;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty())
      continue;
    ids.push_back(std::stoll(tok));
  }
  return ids;
}

// ============================================================
// Minimal Qwen3 BPE tokenizer (ByteLevel, vocab.txt format)
// Ported from buddy::Text<T,N>::tokenizeQwen3 / revertQwen3
// ============================================================

struct Qwen3Tokenizer {
  std::unordered_map<std::string, size_t> tokenToId;
  std::vector<std::string> idToToken;

  void load(const std::string &vocabPath) {
    std::ifstream fin(vocabPath);
    if (!fin.is_open())
      throw std::runtime_error("cannot open vocab: " + vocabPath);
    std::string line;
    size_t idx = 0;
    while (std::getline(fin, line)) {
      tokenToId[line] = idx++;
      idToToken.push_back(line);
    }
  }

  // Convert raw UTF-8 bytes to byte-level BPE string
  // (matches Python's bytes_to_unicode)
  static std::string bytesToBpe(const std::string &input) {
    auto byteToUnicode = [](unsigned char b) -> std::string {
      static const std::unordered_map<unsigned char, std::string> b2u = []() {
        std::unordered_map<unsigned char, std::string> m;
        auto toCodePoint = [](unsigned char b) -> unsigned int {
          if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174))
            return b;
          static const std::vector<unsigned char> extras = []() {
            std::vector<unsigned char> v;
            for (int x = 0; x < 256; ++x)
              if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) ||
                    (x >= 174 && x <= 255)))
                v.push_back((unsigned char)x);
            return v;
          }();
          for (size_t i = 0; i < extras.size(); ++i)
            if (extras[i] == b)
              return 256u + (unsigned int)i;
          return (unsigned int)b;
        };
        for (int i = 0; i < 256; ++i) {
          unsigned int cp = toCodePoint((unsigned char)i);
          std::string s;
          if (cp < 0x80) {
            s += (char)cp;
          } else {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
          }
          m[(unsigned char)i] = s;
        }
        return m;
      }();
      return b2u.at(b);
    };

    std::string result;
    for (unsigned char b : input)
      result += byteToUnicode(b);
    return result;
  }

  // DP tokenization over BPE-encoded string (raw text, no chat template)
  std::vector<int64_t> encodeRaw(const std::string &text) const {
    std::string bpeStr = bytesToBpe(text);
    int n = (int)bpeStr.size();
    std::vector<float> score(n + 1, -1e10f);
    std::vector<size_t> prevId(n + 1, 0);
    std::vector<int> prevPos(n + 1, 0);
    score[0] = 0.0f;

    for (int i = 0; i < n; ++i) {
      if (score[i] < -1e9f)
        continue;
      for (int len = 1; len <= std::min(64, n - i); ++len) {
        std::string sub = bpeStr.substr(i, len);
        auto it = tokenToId.find(sub);
        if (it != tokenToId.end()) {
          float s = score[i] + (float)len * (float)len;
          if (s > score[i + len]) {
            score[i + len] = s;
            prevId[i + len] = it->second;
            prevPos[i + len] = i;
          }
        }
      }
    }

    std::vector<size_t> res;
    int cur = n;
    while (cur > 0) {
      if (score[cur] < -1e9f) {
        --cur;
        continue;
      }
      res.push_back(prevId[cur]);
      cur = prevPos[cur];
    }
    std::vector<int64_t> ids;
    ids.reserve(res.size());
    for (auto it = res.rbegin(); it != res.rend(); ++it)
      ids.push_back((int64_t)*it);
    return ids;
  }

  // Encode with Qwen3 chat template:
  // <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
  std::vector<int64_t> encode(const std::string &text) const {
    // Special token IDs (same as BuddyQwen3)
    const int64_t BOS = 151644; // <|im_start|>
    const int64_t EOS = 151645; // <|im_end|>
    const int64_t USER = 872;   // "user"
    const int64_t ASST = 77091; // "assistant"
    const int64_t NL = 198;     // "\n"

    std::vector<int64_t> ids;
    ids.push_back(BOS);
    ids.push_back(USER);
    ids.push_back(NL);
    for (int64_t id : encodeRaw(text))
      ids.push_back(id);
    ids.push_back(EOS);
    ids.push_back(NL);
    ids.push_back(BOS);
    ids.push_back(ASST);
    ids.push_back(NL);
    return ids;
  }

  // Convert a Unicode code point back to its original byte
  static unsigned char revertBpeChar(unsigned int code) {
    if ((code >= 33 && code <= 126) || (code >= 161 && code <= 172) ||
        (code >= 174 && code <= 255))
      return (unsigned char)code;
    static const std::unordered_map<unsigned int, unsigned char> extraMap =
        []() {
          std::unordered_map<unsigned int, unsigned char> m;
          int n = 0;
          for (int b = 0; b < 256; ++b)
            if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                  (b >= 174 && b <= 255)))
              m[256u + n++] = (unsigned char)b;
          return m;
        }();
    auto it = extraMap.find(code);
    return (it != extraMap.end()) ? it->second : (unsigned char)code;
  }

  std::string decode(const std::vector<int64_t> &ids) const {
    // Stop at <|endoftext|> (151643) or <|im_end|> (151645)
    std::vector<unsigned char> buf;
    for (int64_t id : ids) {
      if (id == 151643 || id == 151645)
        break;
      if (id < 0 || (size_t)id >= idToToken.size())
        continue;
      const std::string &tok = idToToken[(size_t)id];
      for (size_t j = 0; j < tok.size();) {
        unsigned char c = (unsigned char)tok[j];
        unsigned int code = 0;
        int len = 0;
        if (c < 0x80) {
          code = c;
          len = 1;
        } else if ((c & 0xE0) == 0xC0) {
          code = (c & 0x1F) << 6 | ((unsigned char)tok[j + 1] & 0x3F);
          len = 2;
        } else if ((c & 0xF0) == 0xE0) {
          code = (c & 0x0F) << 12 | ((unsigned char)tok[j + 1] & 0x3F) << 6 |
                 ((unsigned char)tok[j + 2] & 0x3F);
          len = 3;
        } else if ((c & 0xF8) == 0xF0) {
          code = (c & 0x07) << 18 | ((unsigned char)tok[j + 1] & 0x3F) << 12 |
                 ((unsigned char)tok[j + 2] & 0x3F) << 6 |
                 ((unsigned char)tok[j + 3] & 0x3F);
          len = 4;
        } else {
          ++j;
          continue;
        }
        j += len;
        buf.push_back(revertBpeChar(code));
      }
    }
    std::string result(buf.begin(), buf.end());
    if (!result.empty() && (unsigned char)result[0] == 32)
      result.erase(0, 1);
    return result;
  }
};

static void buildRope(std::vector<float> &cos, std::vector<float> &sin, int seq,
                      int headDim, int startPos, float ropeTheta) {
  for (int p = 0; p < seq; ++p) {
    int pos = startPos + p;
    for (int i = 0; i < headDim; ++i) {
      int pair = (i < headDim / 2) ? i : (i - headDim / 2);
      float invFreq = std::pow(ropeTheta, -2.0f * pair / headDim);
      float angle = pos * invFreq;
      cos[p * headDim + i] = std::cos(angle);
      sin[p * headDim + i] = std::sin(angle);
    }
  }
}

static int64_t argmax(const std::vector<float> &v) {
  return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

static void rmsNorm(std::vector<float> &out, const std::vector<float> &x,
                    const float *weight, float eps) {
  float meanSq = 0.0f;
  for (size_t i = 0; i < x.size(); ++i)
    meanSq += x[i] * x[i];
  meanSq /= (float)x.size();
  float inv = 1.0f / std::sqrt(meanSq + eps);
  for (size_t i = 0; i < x.size(); ++i)
    out[i] = x[i] * inv * weight[i];
}

int main(int argc, char **argv) {
#ifndef QWEN3_NEXTGPU_BUILD_PATH
#define QWEN3_NEXTGPU_BUILD_PATH "./"
#endif

  const std::string title =
      "Qwen3-0.6B GPU Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  const std::string weightsPath =
      std::string(QWEN3_NEXTGPU_BUILD_PATH) + "qwen3_nextgpu_weights.bin";
  const std::string tokenizerPath =
      std::string(QWEN3_NEXTGPU_BUILD_PATH) + "vocab.txt";
  int maxNew = 200;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--max-new" && i + 1 < argc)
      maxNew = std::stoi(argv[++i]);
  }

  std::cout << "\033[34;1m[Log] \033[0mLoading tokenizer..." << std::endl;
  Qwen3Tokenizer tokenizer;
  tokenizer.load(tokenizerPath);

  std::cout << "\033[34;1m[Log] \033[0mLoading weights..." << std::endl;
  Weights W = loadWeights(weightsPath);
  const int64_t Hidden = W.hidden;
  const int64_t AttnHidden = Hidden * 2;
  constexpr int64_t COMPILED_SEQ = 512; // must match the compiled MLIR kernels
  if ((int64_t)W.seq != COMPILED_SEQ)
    throw std::runtime_error(
        "weights seq_len=" + std::to_string(W.seq) +
        " but kernels were compiled with seq_len=" +
        std::to_string(COMPILED_SEQ) +
        ". Re-export weights with import-qwen3-nextgpu.py");
  const int64_t Seq = COMPILED_SEQ;
  const int64_t HeadNum = W.heads;
  const int64_t HeadDim = Hidden / HeadNum;
  const int64_t FfnHidden = W.ffn;
  const int64_t Vocab = W.vocab;
  const float RopeTheta = 1000000.0f;
  const float RmsEps = 1e-6f;

  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  std::string prompt;
  std::getline(std::cin, prompt);
  std::cout << std::endl;

  std::vector<int64_t> inputIds = tokenizer.encode(prompt);
  if (inputIds.empty())
    throw std::runtime_error("tokenizer returned empty IDs");
  if ((int64_t)inputIds.size() > Seq)
    inputIds.resize((size_t)Seq);

  auto runPrefillFromIds = [&](const std::vector<int64_t> &ids,
                               std::vector<float> &lastHiddenOut,
                               std::vector<std::vector<float>> &kCachesOut,
                               std::vector<std::vector<float>> &vCachesOut) {
    std::vector<float> hiddenSeq((size_t)Seq * (size_t)Hidden, 0.0f);
    for (size_t t = 0; t < ids.size(); ++t) {
      int64_t tok = std::max<int64_t>(0, std::min<int64_t>(ids[t], Vocab - 1));
      const float *emb = W.embed + tok * Hidden;
      std::copy(emb, emb + Hidden, hiddenSeq.data() + t * Hidden);
    }

    std::vector<float> prefillMask((size_t)Seq * (size_t)Seq, -1e9f);
    for (int64_t i = 0; i < Seq; ++i)
      for (int64_t j = 0; j <= i; ++j)
        if (j < (int64_t)ids.size())
          prefillMask[(size_t)i * (size_t)Seq + (size_t)j] = 0.0f;

    std::vector<float> prefillCos((size_t)Seq * (size_t)HeadDim);
    std::vector<float> prefillSin((size_t)Seq * (size_t)HeadDim);
    buildRope(prefillCos, prefillSin, (int)Seq, (int)HeadDim, 0, RopeTheta);

    kCachesOut.assign(
        W.layers, std::vector<float>(
                      (size_t)HeadNum * (size_t)Seq * (size_t)HeadDim, 0.0f));
    vCachesOut.assign(
        W.layers, std::vector<float>(
                      (size_t)HeadNum * (size_t)Seq * (size_t)HeadDim, 0.0f));

    std::vector<float> layerIn = hiddenSeq;
    std::vector<float> layerOut((size_t)Seq * (size_t)Hidden, 0.0f);

    for (uint32_t l = 0; l < W.layers; ++l) {
      auto &L = W.layer[l];

      auto mr_x = makeMemRef<float, 3>(layerIn.data(), {1, Seq, Hidden});
      auto mr_norm1 = makeMemRef<float, 3>(L.norm1, {1, 1, Hidden});
      auto mr_wq = makeMemRef<float, 3>(L.wq, {1, Hidden, AttnHidden});
      auto mr_wk = makeMemRef<float, 3>(L.wk, {1, Hidden, Hidden});
      auto mr_wv = makeMemRef<float, 3>(L.wv, {1, Hidden, Hidden});
      auto mr_qn = makeMemRef<float, 4>(L.q_norm, {1, 1, 1, HeadDim});
      auto mr_kn = makeMemRef<float, 4>(L.k_norm, {1, 1, 1, HeadDim});
      auto mr_wo = makeMemRef<float, 3>(L.wo, {1, AttnHidden, Hidden});
      auto mr_mask = makeMemRef<float, 4>(prefillMask.data(), {1, 1, Seq, Seq});
      auto mr_cos =
          makeMemRef<float, 4>(prefillCos.data(), {1, 1, Seq, HeadDim});
      auto mr_sin =
          makeMemRef<float, 4>(prefillSin.data(), {1, 1, Seq, HeadDim});
      auto mr_norm2 = makeMemRef<float, 3>(L.norm2, {1, 1, Hidden});
      auto mr_gate = makeMemRef<float, 3>(L.ffn_gate, {1, Hidden, FfnHidden});
      auto mr_up = makeMemRef<float, 3>(L.ffn_up, {1, Hidden, FfnHidden});
      auto mr_down = makeMemRef<float, 3>(L.ffn_down, {1, FfnHidden, Hidden});

      PrefillReturn ret{};
      ret.hidden = makeMemRef<float, 3>(layerOut.data(), {1, Seq, Hidden});
      ret.k_cache = makeMemRef<float, 4>(kCachesOut[l].data(),
                                         {1, HeadNum, Seq, HeadDim});
      ret.v_cache = makeMemRef<float, 4>(vCachesOut[l].data(),
                                         {1, HeadNum, Seq, HeadDim});

      _mlir_ciface_qwen3_prefill_kernel(&ret, &mr_x, &mr_norm1, &mr_wq, &mr_wk,
                                        &mr_wv, &mr_qn, &mr_kn, &mr_wo,
                                        &mr_mask, &mr_cos, &mr_sin, &mr_norm2,
                                        &mr_gate, &mr_up, &mr_down);

      float *hout = ret.hidden.data + ret.hidden.offset;
      float *kout = ret.k_cache.data + ret.k_cache.offset;
      float *vout = ret.v_cache.data + ret.v_cache.offset;

      std::copy(hout, hout + (Seq * Hidden), layerOut.data());
      std::copy(kout, kout + (HeadNum * Seq * HeadDim), kCachesOut[l].data());
      std::copy(vout, vout + (HeadNum * Seq * HeadDim), vCachesOut[l].data());
      free(ret.hidden.data);
      free(ret.k_cache.data);
      free(ret.v_cache.data);

      layerIn.swap(layerOut);
    }

    int64_t lastPos = (int64_t)ids.size() - 1;
    lastHiddenOut.assign((size_t)Hidden, 0.0f);
    std::copy(layerIn.begin() + lastPos * Hidden,
              layerIn.begin() + (lastPos + 1) * Hidden, lastHiddenOut.begin());
  };

  std::vector<std::vector<float>> kCaches;
  std::vector<std::vector<float>> vCaches;
  std::vector<float> prefillLastHidden;

  const auto prefillStart = std::chrono::high_resolution_clock::now();
  runPrefillFromIds(inputIds, prefillLastHidden, kCaches, vCaches);
  const auto prefillEnd = std::chrono::high_resolution_clock::now();
  const double prefillSec =
      std::chrono::duration<double>(prefillEnd - prefillStart).count();
  std::cout << "\033[34;1m[Log] \033[0mPrefill: " << prefillSec << "s ("
            << (double)inputIds.size() / prefillSec << " tokens/s)"
            << std::endl;

  std::vector<int64_t> allIds = inputIds;

  auto computeLogits =
      [&](const std::vector<float> &hiddenToken) -> std::vector<float> {
    std::vector<float> normed((size_t)Hidden, 0.0f);
    rmsNorm(normed, hiddenToken, W.final_norm, RmsEps);
    std::vector<float> logits((size_t)Vocab, 0.0f);
    for (int64_t v = 0; v < Vocab; ++v) {
      float acc = 0.0f;
      for (int64_t h = 0; h < Hidden; ++h)
        acc += normed[(size_t)h] *
               W.lm_head[(size_t)h * (size_t)Vocab + (size_t)v];
      logits[(size_t)v] = acc;
    }
    return logits;
  };

  int64_t nextId = argmax(computeLogits(prefillLastHidden));

  double decodeAccumSec = 0.0;
  size_t decodeTokenCount = 0;

  for (int step = 0; step < maxNew; ++step) {
    if ((int64_t)allIds.size() >= Seq)
      break;

    allIds.push_back(nextId);
    int64_t cachePosValue = (int64_t)allIds.size() - 1;

    std::vector<float> tokenHidden((size_t)Hidden, 0.0f);
    const float *emb = W.embed + std::min<int64_t>(nextId, Vocab - 1) * Hidden;
    std::copy(emb, emb + Hidden, tokenHidden.begin());

    std::vector<float> decodeMask((size_t)Seq, -1e9f);
    for (int64_t j = 0; j <= cachePosValue; ++j)
      decodeMask[(size_t)j] = 0.0f;

    std::vector<float> decodeCos((size_t)HeadDim);
    std::vector<float> decodeSin((size_t)HeadDim);
    buildRope(decodeCos, decodeSin, 1, (int)HeadDim, (int)cachePosValue,
              RopeTheta);

    std::vector<int64_t> cachePosBuf{cachePosValue};

    const auto stepStart = std::chrono::high_resolution_clock::now();

    for (uint32_t l = 0; l < W.layers; ++l) {
      auto &L = W.layer[l];

      auto mr_x = makeMemRef<float, 3>(tokenHidden.data(), {1, 1, Hidden});
      auto mr_norm1 = makeMemRef<float, 3>(L.norm1, {1, 1, Hidden});
      auto mr_wq = makeMemRef<float, 3>(L.wq, {1, Hidden, AttnHidden});
      auto mr_wk = makeMemRef<float, 3>(L.wk, {1, Hidden, Hidden});
      auto mr_wv = makeMemRef<float, 3>(L.wv, {1, Hidden, Hidden});
      auto mr_qn = makeMemRef<float, 4>(L.q_norm, {1, 1, 1, HeadDim});
      auto mr_kn = makeMemRef<float, 4>(L.k_norm, {1, 1, 1, HeadDim});
      auto mr_wo = makeMemRef<float, 3>(L.wo, {1, AttnHidden, Hidden});
      auto mr_mask = makeMemRef<float, 4>(decodeMask.data(), {1, 1, 1, Seq});
      auto mr_cos = makeMemRef<float, 4>(decodeCos.data(), {1, 1, 1, HeadDim});
      auto mr_sin = makeMemRef<float, 4>(decodeSin.data(), {1, 1, 1, HeadDim});
      auto mr_pos = makeMemRef<int64_t, 1>(cachePosBuf.data(), {1});
      auto mr_k =
          makeMemRef<float, 4>(kCaches[l].data(), {1, HeadNum, Seq, HeadDim});
      auto mr_v =
          makeMemRef<float, 4>(vCaches[l].data(), {1, HeadNum, Seq, HeadDim});
      auto mr_norm2 = makeMemRef<float, 3>(L.norm2, {1, 1, Hidden});
      auto mr_gate = makeMemRef<float, 3>(L.ffn_gate, {1, Hidden, FfnHidden});
      auto mr_up = makeMemRef<float, 3>(L.ffn_up, {1, Hidden, FfnHidden});
      auto mr_down = makeMemRef<float, 3>(L.ffn_down, {1, FfnHidden, Hidden});

      std::vector<float> outHidden((size_t)Hidden, 0.0f);
      DecodeReturn ret{};
      ret.hidden = makeMemRef<float, 3>(outHidden.data(), {1, 1, Hidden});
      ret.k_cache =
          makeMemRef<float, 4>(kCaches[l].data(), {1, HeadNum, Seq, HeadDim});
      ret.v_cache =
          makeMemRef<float, 4>(vCaches[l].data(), {1, HeadNum, Seq, HeadDim});

      _mlir_ciface_qwen3_decode_kernel(&ret, &mr_x, &mr_norm1, &mr_wq, &mr_wk,
                                       &mr_wv, &mr_qn, &mr_kn, &mr_wo, &mr_mask,
                                       &mr_cos, &mr_sin, &mr_pos, &mr_k, &mr_v,
                                       &mr_norm2, &mr_gate, &mr_up, &mr_down);

      float *hout = ret.hidden.data + ret.hidden.offset;
      float *kout = ret.k_cache.data + ret.k_cache.offset;
      float *vout = ret.v_cache.data + ret.v_cache.offset;

      std::copy(hout, hout + Hidden, tokenHidden.data());
      std::copy(kout, kout + (HeadNum * Seq * HeadDim), kCaches[l].data());
      std::copy(vout, vout + (HeadNum * Seq * HeadDim), vCaches[l].data());
      free(ret.hidden.data);
      free(ret.k_cache.data);
      free(ret.v_cache.data);
    }

    nextId = argmax(computeLogits(tokenHidden));

    const auto stepEnd = std::chrono::high_resolution_clock::now();
    const double stepSec =
        std::chrono::duration<double>(stepEnd - stepStart).count();
    decodeAccumSec += stepSec;
    ++decodeTokenCount;

    std::string tokStr = tokenizer.decode({nextId});
    std::cout << "\033[32;1m[Iteration " << step + 1 << "]\033[0m "
              << "Token: " << tokStr << " | Time: " << stepSec << "s"
              << std::endl;

    // Stop at <|im_end|> (151645) or <|endoftext|> (151643)
    if (nextId == 151645 || nextId == 151643)
      break;
  }

  const double decodeTPS =
      decodeTokenCount > 0 ? (double)decodeTokenCount / decodeAccumSec : 0.0;
  std::cout << "\n\033[33;1m[Prefilling]\033[0m "
            << (double)inputIds.size() / prefillSec << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTPS << " tokens/s"
            << std::endl;

  std::vector<int64_t> genIds(allIds.begin() + (ptrdiff_t)inputIds.size(),
                              allIds.end());
  std::cout << "\033[33;1m[Output]\033[0m " << tokenizer.decode(genIds)
            << std::endl;

  std::cout.flush();
  // Use _Exit to skip MLIR CUDA runtime's atexit handlers which attempt to
  // call cuModuleUnload after the CUDA context has already been destroyed.
  std::_Exit(0);
}
