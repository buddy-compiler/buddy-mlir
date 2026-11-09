//===- YoloV8Runner.cpp - YOLOv8 detection / pose runner ------------------===//
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
// Pre- and post-processing follow Ultralytics predict of detection and pose
// models at a fixed square input: LetterBox(auto=False, center=True) with pad
// value 114 and bilinear resize, RGB scaled to [0, 1]; non_max_suppression
// (best class per anchor, class-aware greedy NMS); scale_boxes / scale_coords
// of the boxes and keypoints back to the original image.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/YoloV8Runner.h"
#include "buddy/Core/Container.h"
#include "buddy/runtime/core/ModelManifest.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<float, 4> *);

constexpr float PadValue = 114.0f;

struct Options {
  size_t imgSize = 320;
  size_t numClasses = 80;
  size_t numKeypoints = 0;
  size_t numAnchors = 2100;
  float confThreshold = 0.25f;
  float iouThreshold = 0.7f;
  size_t maxDetections = 300;
};

struct Letterbox {
  float gain = 1.0f;
  int padLeft = 0;
  int padTop = 0;
  int width = 0;
  int height = 0;
};

struct Keypoint {
  float x, y, visibility;
};

struct Detection {
  float x1, y1, x2, y2;
  float score;
  int classId;
  size_t anchor;
  std::vector<Keypoint> keypoints;
};

double secondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

std::string attr(const ModelManifest &manifest, const char *key) {
  auto it = manifest.moduleAttrs.find(key);
  return it == manifest.moduleAttrs.end() ? std::string() : it->second;
}

size_t sizeAttr(const ModelManifest &manifest, const char *key,
                size_t fallback) {
  const std::string value = attr(manifest, key);
  return value.empty() ? fallback : static_cast<size_t>(std::stoull(value));
}

float floatAttr(const ModelManifest &manifest, const char *key,
                float fallback) {
  const std::string value = attr(manifest, key);
  return value.empty() ? fallback : std::stof(value);
}

std::vector<std::string> loadLabels(const std::string &path, size_t count) {
  std::vector<std::string> labels;
  if (!path.empty()) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line))
      if (!line.empty())
        labels.push_back(line);
  }
  for (size_t i = labels.size(); i < count; ++i)
    labels.push_back("class" + std::to_string(i));
  return labels;
}

void loadWeights(const std::string &path, MemRef<float, 1> &params) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("YoloV8Runner: failed to open weights: " + path);
  input.read(reinterpret_cast<char *>(params.getData()),
             static_cast<std::streamsize>(sizeof(float) * params.getSize()));
  if (!input)
    throw std::runtime_error("YoloV8Runner: error reading weights: " + path);
}

/// Decodes `path` and letterboxes it into the NCHW input of the model.
Letterbox preprocess(const std::string &path, size_t imgSize,
                     MemRef<float, 4> &input) {
  int w = 0, h = 0, channels = 0;
  unsigned char *rgb = stbi_load(path.c_str(), &w, &h, &channels, 3);
  if (!rgb)
    throw std::runtime_error("YoloV8Runner: cannot decode image " + path +
                             ": " + stbi_failure_reason());

  const int size = static_cast<int>(imgSize);
  Letterbox lb;
  lb.width = w;
  lb.height = h;
  lb.gain =
      std::min(static_cast<float>(size) / h, static_cast<float>(size) / w);
  const int newW = static_cast<int>(std::lround(w * lb.gain));
  const int newH = static_cast<int>(std::lround(h * lb.gain));
  lb.padLeft = static_cast<int>(std::lround((size - newW) / 2.0 - 0.1));
  lb.padTop = static_cast<int>(std::lround((size - newH) / 2.0 - 0.1));

  float *dst = input.getData();
  std::fill(dst, dst + 3 * imgSize * imgSize, PadValue / 255.0f);
  // Bilinear with half-pixel centers, as cv2.resize(INTER_LINEAR), rounded
  // to 8 bits as the resized uint8 image is.
  const float sx = static_cast<float>(w) / newW;
  const float sy = static_cast<float>(h) / newH;
  std::vector<int> x0(newW), x1(newW);
  std::vector<float> fx(newW);
  for (int x = 0; x < newW; ++x) {
    float src = std::max((x + 0.5f) * sx - 0.5f, 0.0f);
    x0[x] = std::min(static_cast<int>(src), w - 1);
    x1[x] = std::min(x0[x] + 1, w - 1);
    fx[x] = src - x0[x];
  }
  const size_t plane = imgSize * imgSize;
  for (int y = 0; y < newH; ++y) {
    float src = std::max((y + 0.5f) * sy - 0.5f, 0.0f);
    const int y0 = std::min(static_cast<int>(src), h - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const float fy = src - y0;
    const unsigned char *r0 = rgb + static_cast<size_t>(y0) * w * 3;
    const unsigned char *r1 = rgb + static_cast<size_t>(y1) * w * 3;
    float *row =
        dst + static_cast<size_t>(y + lb.padTop) * imgSize + lb.padLeft;
    for (int x = 0; x < newW; ++x) {
      for (int c = 0; c < 3; ++c) {
        const float top =
            r0[x0[x] * 3 + c] * (1 - fx[x]) + r0[x1[x] * 3 + c] * fx[x];
        const float bot =
            r1[x0[x] * 3 + c] * (1 - fx[x]) + r1[x1[x] * 3 + c] * fx[x];
        row[c * plane + x] = std::round(top * (1 - fy) + bot * fy) / 255.0f;
      }
    }
  }
  stbi_image_free(rgb);
  return lb;
}

float iou(const Detection &a, const Detection &b) {
  const float iw = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float ih = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = iw * ih;
  const float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

std::vector<Detection> postprocess(MemRef<float, 3> &output, const Options &opt,
                                   const Letterbox &lb) {
  const float *data = output.getData();
  const intptr_t *strides = output.getStrides();
  auto at = [&](size_t row, size_t anchor) {
    return data[row * strides[1] + anchor * strides[2]];
  };

  std::vector<Detection> candidates;
  for (size_t a = 0; a < opt.numAnchors; ++a) {
    size_t best = 0;
    float score = at(4, a);
    for (size_t c = 1; c < opt.numClasses; ++c) {
      const float s = at(4 + c, a);
      if (s > score) {
        score = s;
        best = c;
      }
    }
    if (!(score > opt.confThreshold))
      continue;
    const float cx = at(0, a), cy = at(1, a);
    const float hw = at(2, a) / 2, hh = at(3, a) / 2;
    candidates.push_back({cx - hw,
                          cy - hh,
                          cx + hw,
                          cy + hh,
                          score,
                          static_cast<int>(best),
                          a,
                          {}});
  }
  std::stable_sort(
      candidates.begin(), candidates.end(),
      [](const Detection &a, const Detection &b) { return a.score > b.score; });

  std::vector<Detection> kept;
  std::vector<bool> removed(candidates.size(), false);
  for (size_t i = 0; i < candidates.size() && kept.size() < opt.maxDetections;
       ++i) {
    if (removed[i])
      continue;
    kept.push_back(candidates[i]);
    for (size_t j = i + 1; j < candidates.size(); ++j)
      if (!removed[j] && candidates[j].classId == candidates[i].classId &&
          iou(candidates[i], candidates[j]) > opt.iouThreshold)
        removed[j] = true;
  }

  const size_t kptRow = 4 + opt.numClasses;
  for (Detection &d : kept) {
    for (size_t k = 0; k < opt.numKeypoints; ++k) {
      const float x = at(kptRow + 3 * k, d.anchor);
      const float y = at(kptRow + 3 * k + 1, d.anchor);
      d.keypoints.push_back(
          {std::clamp((x - lb.padLeft) / lb.gain, 0.0f, float(lb.width)),
           std::clamp((y - lb.padTop) / lb.gain, 0.0f, float(lb.height)),
           at(kptRow + 3 * k + 2, d.anchor)});
    }
    d.x1 = std::clamp((d.x1 - lb.padLeft) / lb.gain, 0.0f, float(lb.width));
    d.x2 = std::clamp((d.x2 - lb.padLeft) / lb.gain, 0.0f, float(lb.width));
    d.y1 = std::clamp((d.y1 - lb.padTop) / lb.gain, 0.0f, float(lb.height));
    d.y2 = std::clamp((d.y2 - lb.padTop) / lb.gain, 0.0f, float(lb.height));
  }
  return kept;
}

} // namespace

void YoloV8Runner::run(const RunConfig &cfg) {
  if (cfg.imagePath.empty())
    throw std::runtime_error("YoloV8Runner: pass --image <path>");
#if defined(__GLIBC__)
  // Every kernel of the model allocates its output (up to a few MB) and the
  // model frees it after its last use. By default glibc maps each buffer
  // above 128 KB afresh and unmaps it when freed, so every forward call
  // faulted its pages in again (~7000 page faults per 320x320 pose frame).
  // Served from the heap and not trimmed, the buffers of one call reuse the
  // pages of the previous one.
  mallopt(M_MMAP_THRESHOLD, 32 << 20);
  mallopt(M_TRIM_THRESHOLD, 1 << 30);
#endif

  // The model library is compiled for one variant: its parameter count,
  // input size and outputs. The .rax manifest says which; a library and a
  // weight file passed on their own say nothing, and are not taken.
  if (cfg.raxPath.empty())
    throw std::runtime_error(
        "YoloV8Runner: pass --model <.rax> (the model library is run with "
        "the parameters and options of its manifest)");
  ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
  std::string soPath = manifest.soPath;
  std::string weightsPath;
  for (const auto &constant : manifest.constants)
    if (constant.name == "params")
      weightsPath = constant.path;
  if (weightsPath.empty() && !manifest.weightPaths.empty())
    weightsPath = manifest.weightPaths.front();
  const std::string labelsPath = manifest.vocabPath;
  const std::vector<std::string> deps = manifest.dependentSoPaths;
  const std::string modelName =
      manifest.modelName.empty() ? "yolov8" : manifest.modelName;
  Options opt;
  opt.imgSize = sizeAttr(manifest, "img_size", opt.imgSize);
  opt.numClasses = sizeAttr(manifest, "num_classes", opt.numClasses);
  opt.numKeypoints = sizeAttr(manifest, "num_keypoints", opt.numKeypoints);
  opt.numAnchors = sizeAttr(manifest, "num_anchors", opt.numAnchors);
  opt.confThreshold = floatAttr(manifest, "conf_threshold", opt.confThreshold);
  opt.iouThreshold = floatAttr(manifest, "iou_threshold", opt.iouThreshold);
  opt.maxDetections = sizeAttr(manifest, "max_detections", opt.maxDetections);
  // The f32 slots the compiled model reads its parameters from.
  const size_t paramsSize = sizeAttr(manifest, "params_size", 0);
  if (soPath.empty() || weightsPath.empty() || paramsSize == 0)
    throw std::runtime_error(
        "YoloV8Runner: the manifest of " + cfg.raxPath +
        " lacks the model library, the weights or params_size (rebuild it)");

  for (const std::string &dep : deps)
    if (!dlopen(dep.c_str(), RTLD_NOW | RTLD_GLOBAL))
      throw std::runtime_error(
          "YoloV8Runner: dlopen dependency failed: " + dep + ": " + dlerror());
  void *model = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!model)
    throw std::runtime_error("YoloV8Runner: dlopen model failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(model, "_mlir_ciface_forward"));
  if (const char *err = dlerror())
    throw std::runtime_error("YoloV8Runner: missing _mlir_ciface_forward: " +
                             std::string(err));

  const auto loadStart = std::chrono::steady_clock::now();
  // The model reads paramsSize f32 at fixed offsets: a shorter file would
  // have it read past the buffer.
  const uintmax_t bytes = std::filesystem::file_size(weightsPath);
  if (bytes != paramsSize * sizeof(float))
    throw std::runtime_error(
        "YoloV8Runner: " + weightsPath + " has " + std::to_string(bytes) +
        " bytes, the model reads " + std::to_string(paramsSize) + " f32");
  MemRef<float, 1> params(std::vector<size_t>{paramsSize});
  loadWeights(weightsPath, params);
  const double loadSeconds = secondsSince(loadStart);
  const std::vector<std::string> labels =
      loadLabels(labelsPath, opt.numClasses);

  const auto preStart = std::chrono::steady_clock::now();
  MemRef<float, 4> input(std::vector<size_t>{1, 3, opt.imgSize, opt.imgSize});
  const Letterbox lb = preprocess(cfg.imagePath, opt.imgSize, input);
  const double preSeconds = secondsSince(preStart);

  // BUDDY_YOLOV8_RUNS=N repeats the forward pass to time a warm model.
  int runs = 1;
  if (const char *env = std::getenv("BUDDY_YOLOV8_RUNS"))
    runs = std::max(1, std::atoi(env));
  std::vector<double> inferSeconds;
  MemRef<float, 3> output(std::vector<size_t>{1, 1, 1}, false, 0);
  for (int r = 0; r < runs; ++r) {
    if (r)
      free(output.release());
    const auto start = std::chrono::steady_clock::now();
    forward(&output, &params, &input);
    inferSeconds.push_back(secondsSince(start));
  }
  const intptr_t *sizes = output.getSizes();
  if (!output.getData() ||
      sizes[1] != intptr_t(4 + opt.numClasses + 3 * opt.numKeypoints) ||
      sizes[2] != intptr_t(opt.numAnchors))
    throw std::runtime_error("YoloV8Runner: unexpected output shape");

  const auto postStart = std::chrono::steady_clock::now();
  const std::vector<Detection> detections = postprocess(output, opt, lb);
  const double postSeconds = secondsSince(postStart);
  free(output.release());

  if (!cfg.suppressStats) {
    std::cerr << "\033[33;1mYOLOv8 "
              << (opt.numKeypoints ? "Pose Estimation" : "Object Detection")
              << " Powered by Buddy Compiler\033[0m\n";
    std::cerr << "  model: " << modelName << " (" << opt.imgSize << "x"
              << opt.imgSize << ")\n";
    std::cerr << "  image: " << cfg.imagePath << " (" << lb.width << "x"
              << lb.height << ")\n";
    std::cerr << "  weights load: " << loadSeconds * 1e3 << " ms\n";
    std::cerr << "  preprocess:   " << preSeconds * 1e3 << " ms\n";
    std::cerr << "  inference:    " << inferSeconds.front() * 1e3
              << " ms (first run)\n";
    if (runs > 1) {
      std::vector<double> warm(inferSeconds.begin() + 1, inferSeconds.end());
      std::sort(warm.begin(), warm.end());
      const double mean =
          std::accumulate(warm.begin(), warm.end(), 0.0) / warm.size();
      std::cerr << "  inference:    " << mean * 1e3 << " ms mean, "
                << warm[warm.size() / 2] * 1e3 << " ms median, "
                << warm.front() * 1e3 << " ms min over " << warm.size()
                << " warm runs\n";
    }
    std::cerr << "  postprocess:  " << postSeconds * 1e3 << " ms\n";
  }

  for (size_t i = 0; i < detections.size(); ++i) {
    const Detection &d = detections[i];
    char line[256];
    std::snprintf(line, sizeof(line),
                  "[%zu] class_id=%d label=%s score=%.4f box=(%.1f, %.1f, "
                  "%.1f, %.1f)",
                  i, d.classId, labels[d.classId].c_str(), d.score, d.x1, d.y1,
                  d.x2, d.y2);
    std::cout << line << "\n";
    if (d.keypoints.empty())
      continue;
    std::cout << "    keypoints=[";
    for (size_t k = 0; k < d.keypoints.size(); ++k) {
      const Keypoint &kp = d.keypoints[k];
      std::snprintf(line, sizeof(line), "%s(%.1f, %.1f, %.2f)", k ? ", " : "",
                    kp.x, kp.y, kp.visibility);
      std::cout << line;
    }
    std::cout << "]\n";
  }
  std::cout << "Detections: " << detections.size() << std::endl;
  // The model library and its dependencies stay loaded until the process
  // exits: the threads of the model thread pool run in them.
}

} // namespace runtime
} // namespace buddy
