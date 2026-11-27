//===- PaddleocrImagePreprocess.h - Fixed-grid PaddleOCR image input ------===//
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

#ifndef BUDDY_MODELS_PADDLEOCR_IMAGE_PREPROCESS_H
#define BUDDY_MODELS_PADDLEOCR_IMAGE_PREPROCESS_H

#include "../qwen3_vl/ImagePreprocess.h"

namespace buddy::runtime::paddleocr_image {

// The compiled graph has a fixed [gridH, gridW] input. Use the same Pillow
// bicubic resizer as the Qwen3-VL runner, then pack raster-order CHW patches.
inline void preprocessImage(const std::string &imagePath, int gridH, int gridW,
                            int patchSize, std::vector<float> &pixelValues) {
  int width = 0, height = 0, channels = 0;
  uint8_t *decoded =
      stbi_load(imagePath.c_str(), &width, &height, &channels, 3);
  if (!decoded)
    throw std::runtime_error("paddleocr: failed to decode image: " + imagePath);

  qwen3vl_image::detail::RGBImage source;
  source.width = width;
  source.height = height;
  source.data.assign(decoded,
                     decoded + static_cast<size_t>(width) * height * 3);
  stbi_image_free(decoded);
  auto resized = qwen3vl_image::detail::resizeBicubicPillow(
      source, gridW * patchSize, gridH * patchSize);

  const size_t patchPixels = static_cast<size_t>(patchSize) * patchSize;
  pixelValues.resize(static_cast<size_t>(gridH) * gridW * 3 * patchPixels);
  for (int patchRow = 0; patchRow < gridH; ++patchRow) {
    for (int patchCol = 0; patchCol < gridW; ++patchCol) {
      const size_t patchIndex =
          static_cast<size_t>(patchRow) * gridW + patchCol;
      for (int channel = 0; channel < 3; ++channel) {
        float *plane =
            pixelValues.data() + (patchIndex * 3 + channel) * patchPixels;
        for (int row = 0; row < patchSize; ++row) {
          const uint8_t *pixels =
              resized.pixel(patchCol * patchSize, patchRow * patchSize + row);
          for (int col = 0; col < patchSize; ++col)
            plane[row * patchSize + col] =
                static_cast<float>(pixels[col * 3 + channel]) / 127.5f - 1.0f;
        }
      }
    }
  }
}

} // namespace buddy::runtime::paddleocr_image

#endif // BUDDY_MODELS_PADDLEOCR_IMAGE_PREPROCESS_H
