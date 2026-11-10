//===- YoloV8Runner.h - YOLOv8 detection / pose runner --------------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_YOLOV8RUNNER_H
#define BUDDY_RUNTIME_MODELS_YOLOV8RUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// Object detection and pose estimation runner for YOLOv8.
///
/// The compiled kernel produces the raw head output
/// [1, 4 + classes + 3 * keypoints, anchors]. The runner owns image decoding,
/// letterboxing, weight loading, kernel invocation, the confidence filter,
/// class-aware NMS, and mapping the boxes and keypoints back to the original
/// image, as Ultralytics predicts.
class YoloV8Runner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_YOLOV8RUNNER_H
