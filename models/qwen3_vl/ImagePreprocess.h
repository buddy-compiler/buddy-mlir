//===- ImagePreprocess.h - Pure C++ Qwen3-VL image preprocessing ---------===//
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
// Reproduces, in pure C++, the exact tensor that
// codegen/qwen3_vl_codegen.py's cmd_preprocess() extracts for pixel_values:
//
//   PIL.Image.open(path).convert("RGB").resize((448, 224))
//     -> Qwen2VLImageProcessorFast.preprocess(...)
//
// For the pinned canonical size (448x224 = 14x28 patches, both already
// multiples of patch_size*merge_size=32), the HF processor's internal
// `smart_resize` is a no-op, so the *only* resize that ever actually runs is
// the PIL `.resize()` call above. That makes an exact Pillow-bicubic port
// (not merely a "good enough" resize) the correct target, not just a nice
// to have.
//
// - Image decode: stb_image.h (thirdparty/include), vendored from
//   llama.cpp's tools/mtmd, same as its multimodal image loading path.
// - Bicubic resize: ported from llama.cpp's tools/mtmd/mtmd-image.cpp
//   `resize_bicubic_pillow`, itself adapted from Pillow's
//   src/libImaging/Resample.c (filter a=-0.5, separable two-pass,
//   fixed-point accumulation) to bit-match `Image.resize()`.
// - Normalize: image_mean = image_std = 0.5 for all channels (see
//   preprocessor_config.json), i.e. v = pixel/255*2 - 1.
// - Patchify: mirrors Qwen2VLImageProcessor._preprocess's view/permute/
//   reshape (patch_size=16, temporal_patch_size=2, merge_size=2). A static
//   image has no temporal axis, so the HF preprocessor duplicates the single
//   frame across both temporal slots; this implementation just writes the
//   same pixel twice instead of allocating the duplicate frame.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MODELS_QWEN3_VL_IMAGE_PREPROCESS_H
#define BUDDY_MODELS_QWEN3_VL_IMAGE_PREPROCESS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// The including .cpp must `#define STB_IMAGE_IMPLEMENTATION` before its own
// `#include "stb_image.h"` (once, in exactly one translation unit) per
// stb_image's documented usage; this header only calls its decode API.
#include "stb_image.h"

namespace buddy {
namespace runtime {
namespace qwen3vl_image {

namespace detail {

// Interleaved 8-bit RGB image, row-major.
struct RGBImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> data;

  const uint8_t *pixel(int x, int y) const {
    return &data[(static_cast<size_t>(y) * width + x) * 3];
  }
  void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    uint8_t *p = &data[(static_cast<size_t>(y) * width + x) * 3];
    p[0] = r;
    p[1] = g;
    p[2] = b;
  }
};

// Bicubic filter, a = -0.5 (Pillow/Catmull-Rom variant; PyTorch/GGML use
// a = -0.75, which is a different curve).
inline double bicubicFilter(double x) {
  constexpr double a = -0.5;
  if (x < 0.0)
    x = -x;
  if (x < 1.0)
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0)
    return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

inline uint8_t clip8(int32_t val) {
  if (val < 0)
    return 0;
  if (val > 255)
    return 255;
  return static_cast<uint8_t>(val);
}

// Fixed-point precision: 32 (int32_t) - 8 (uint8_t pixels) - 2 (headroom).
constexpr int kPrecisionBits = 32 - 8 - 2;
constexpr double kFilterSupport = 2.0;

// Precomputes 1-D filter coefficients for resizing inSize -> outSize,
// matching Pillow's ImagingResample precompute_coeffs. Returns the kernel
// size; fills `bounds` (per-output [xmin, count]) and `weights` (per-output
// fixed-point taps).
inline int precomputeWeights(int inSize, int outSize, std::vector<int> &bounds,
                             std::vector<int32_t> &weights) {
  double scale = static_cast<double>(inSize) / outSize;
  double filterscale = scale < 1.0 ? 1.0 : scale;
  double support = kFilterSupport * filterscale;
  int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

  std::vector<double> preWeights(static_cast<size_t>(outSize) * ksize);
  bounds.resize(static_cast<size_t>(outSize) * 2);

  for (int xx = 0; xx < outSize; xx++) {
    double center = (xx + 0.5) * scale;
    double ss = 1.0 / filterscale;

    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0)
      xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > inSize)
      xmax = inSize;
    xmax -= xmin;

    double ww = 0.0;
    int x = 0;
    for (; x < xmax; x++) {
      double w = bicubicFilter((x + xmin - center + 0.5) * ss);
      preWeights[static_cast<size_t>(xx) * ksize + x] = w;
      ww += w;
    }
    for (x = 0; x < xmax; x++)
      if (ww != 0.0)
        preWeights[static_cast<size_t>(xx) * ksize + x] /= ww;
    for (; x < ksize; x++)
      preWeights[static_cast<size_t>(xx) * ksize + x] = 0;

    bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
    bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
  }

  weights.resize(static_cast<size_t>(outSize) * ksize);
  const double fxpScale = std::ldexp(1.0, kPrecisionBits);
  for (size_t i = 0; i < weights.size(); i++) {
    double tmp = preWeights[i] * fxpScale;
    tmp += (preWeights[i] < 0) ? -0.5 : 0.5;
    tmp = std::round(tmp);
    tmp = std::clamp(tmp,
                     static_cast<double>(std::numeric_limits<int32_t>::min()),
                     static_cast<double>(std::numeric_limits<int32_t>::max()));
    weights[i] = static_cast<int32_t>(tmp);
  }
  return ksize;
}

inline void resampleHorizontal(const RGBImage &in, RGBImage &out, int outW,
                               int ksize, const std::vector<int> &bounds,
                               const std::vector<int32_t> &weights) {
  out.width = outW;
  out.height = in.height;
  out.data.assign(static_cast<size_t>(outW) * in.height * 3, 0);
  for (int yy = 0; yy < in.height; yy++) {
    for (int xx = 0; xx < outW; xx++) {
      int xmin = bounds[static_cast<size_t>(xx) * 2 + 0];
      int xcnt = bounds[static_cast<size_t>(xx) * 2 + 1];
      int32_t ss0 = 1 << (kPrecisionBits - 1);
      int32_t ss1 = ss0, ss2 = ss0;
      for (int x = 0; x < xcnt; x++) {
        const uint8_t *p = in.pixel(x + xmin, yy);
        int32_t w = weights[static_cast<size_t>(xx) * ksize + x];
        ss0 += p[0] * w;
        ss1 += p[1] * w;
        ss2 += p[2] * w;
      }
      out.setPixel(xx, yy, clip8(ss0 >> kPrecisionBits),
                   clip8(ss1 >> kPrecisionBits), clip8(ss2 >> kPrecisionBits));
    }
  }
}

inline void resampleVertical(const RGBImage &in, RGBImage &out, int outH,
                             int ksize, const std::vector<int> &bounds,
                             const std::vector<int32_t> &weights) {
  out.width = in.width;
  out.height = outH;
  out.data.assign(static_cast<size_t>(in.width) * outH * 3, 0);
  for (int yy = 0; yy < outH; yy++) {
    int ymin = bounds[static_cast<size_t>(yy) * 2 + 0];
    int ycnt = bounds[static_cast<size_t>(yy) * 2 + 1];
    for (int xx = 0; xx < in.width; xx++) {
      int32_t ss0 = 1 << (kPrecisionBits - 1);
      int32_t ss1 = ss0, ss2 = ss0;
      for (int y = 0; y < ycnt; y++) {
        const uint8_t *p = in.pixel(xx, y + ymin);
        int32_t w = weights[static_cast<size_t>(yy) * ksize + y];
        ss0 += p[0] * w;
        ss1 += p[1] * w;
        ss2 += p[2] * w;
      }
      out.setPixel(xx, yy, clip8(ss0 >> kPrecisionBits),
                   clip8(ss1 >> kPrecisionBits), clip8(ss2 >> kPrecisionBits));
    }
  }
}

// Pillow-exact bicubic resize (separable two-pass, horizontal then
// vertical), matching PIL.Image.resize()'s default BICUBIC resample.
inline RGBImage resizeBicubicPillow(const RGBImage &src, int dstW, int dstH) {
  bool needH = dstW != src.width;
  bool needV = dstH != src.height;
  if (!needH && !needV)
    return src;

  std::vector<int> boundsH, boundsV;
  std::vector<int32_t> weightsH, weightsV;
  int ksizeH = 0, ksizeV = 0;
  if (needH)
    ksizeH = precomputeWeights(src.width, dstW, boundsH, weightsH);
  if (needV)
    ksizeV = precomputeWeights(src.height, dstH, boundsV, weightsV);

  RGBImage out;
  if (needH && needV) {
    RGBImage tmp;
    resampleHorizontal(src, tmp, dstW, ksizeH, boundsH, weightsH);
    resampleVertical(tmp, out, dstH, ksizeV, boundsV, weightsV);
  } else if (needH) {
    resampleHorizontal(src, out, dstW, ksizeH, boundsH, weightsH);
  } else {
    resampleVertical(src, out, dstH, ksizeV, boundsV, weightsV);
  }
  return out;
}

} // namespace detail

// Decodes `imagePath`, resizes to the pinned Qwen3-VL vision grid
// (448 wide x 224 tall), normalizes with mean=std=0.5, and patchifies into
// the [392, 1536] layout the compiled vision encoder expects. Throws
// std::runtime_error on decode failure.
inline void preprocessImage(const std::string &imagePath,
                            std::vector<float> &pixelValues) {
  constexpr int kCanonW = 448, kCanonH = 224;
  constexpr int kPatchSize = 16, kMergeSize = 2, kTemporalPatchSize = 2;
  constexpr int kGridH = kCanonH / kPatchSize; // 14
  constexpr int kGridW = kCanonW / kPatchSize; // 28
  constexpr int kNumPatches = kGridH * kGridW; // 392
  constexpr int kPatchDim =
      3 * kTemporalPatchSize * kPatchSize * kPatchSize; // 1536

  int w = 0, h = 0, channels = 0;
  uint8_t *decoded = stbi_load(imagePath.c_str(), &w, &h, &channels, 3);
  if (!decoded)
    throw std::runtime_error("qwen3_vl: failed to decode image: " + imagePath +
                             " (" + stbi_failure_reason() + ")");

  detail::RGBImage src;
  src.width = w;
  src.height = h;
  src.data.assign(decoded, decoded + static_cast<size_t>(w) * h * 3);
  stbi_image_free(decoded);

  detail::RGBImage resized = detail::resizeBicubicPillow(src, kCanonW, kCanonH);

  pixelValues.assign(static_cast<size_t>(kNumPatches) * kPatchDim, 0.0f);
  const int hOuterN = kGridH / kMergeSize; // 7
  const int wOuterN = kGridW / kMergeSize; // 14
  for (int hOuter = 0; hOuter < hOuterN; hOuter++) {
    for (int wOuter = 0; wOuter < wOuterN; wOuter++) {
      for (int hMerge = 0; hMerge < kMergeSize; hMerge++) {
        for (int wMerge = 0; wMerge < kMergeSize; wMerge++) {
          int patchIdx =
              ((hOuter * wOuterN + wOuter) * kMergeSize + hMerge) * kMergeSize +
              wMerge;
          float *dst =
              pixelValues.data() + static_cast<size_t>(patchIdx) * kPatchDim;
          int rowBase = hOuter * kMergeSize * kPatchSize + hMerge * kPatchSize;
          int colBase = wOuter * kMergeSize * kPatchSize + wMerge * kPatchSize;
          for (int c = 0; c < 3; c++) {
            for (int t = 0; t < kTemporalPatchSize; t++) {
              float *plane =
                  dst + (static_cast<size_t>(c) * kTemporalPatchSize + t) *
                            kPatchSize * kPatchSize;
              for (int hp = 0; hp < kPatchSize; hp++) {
                const uint8_t *row = resized.pixel(colBase, rowBase + hp);
                for (int wp = 0; wp < kPatchSize; wp++) {
                  uint8_t v = row[wp * 3 + c];
                  plane[hp * kPatchSize + wp] =
                      static_cast<float>(v) / 127.5f - 1.0f;
                }
              }
            }
          }
        }
      }
    }
  }
}

} // namespace qwen3vl_image
} // namespace runtime
} // namespace buddy

#endif // BUDDY_MODELS_QWEN3_VL_IMAGE_PREPROCESS_H
