# YOLOv8 (pose / detect)

Ultralytics YOLOv8 at a fixed square input, served through the `buddy-cli` /
`.rax` runtime. The build imports the PyTorch checkpoint, compiles the forward
graph to MLIR, links `yolov8_model.so`, and packs `yolov8.rax` with the
`yolov8_runner.so` plugin. The runner decodes the image (JPEG / PNG / BMP,
`stb_image`), letterboxes it, runs the model, and does the confidence filter,
class-aware NMS and the mapping of boxes and keypoints back to the image, as
Ultralytics predicts with `LetterBox(auto=False)`.

| Spec | Model | Input | Output |
| --- | --- | --- | --- |
| `specs/k3_pose_320.json` (default) | `yolov8n-pose.pt` | 320x320 | person boxes + 17 COCO keypoints |
| `specs/k3_detect_320.json` | `yolov8n.pt` | 320x320 | boxes of the 80 COCO classes |

The weights come from Ultralytics under AGPL-3.0, and the built `.rax` and
`arg0.data` contain them: distributing the build outputs, or serving them over
a network, needs AGPL-3.0 compliance or an Ultralytics commercial license. No
Ultralytics code is in this directory (the letterbox, NMS and decode are
reimplemented); `ultralytics` is only used at build time to load the checkpoint.

## Build for the SpacemiT K3

Needs the Python environment `buddy-mlir` was built with, plus `ultralytics`
(build time only), and the cross-compilation setup of
[docs/CrossCompilingRaxFile.md](../../docs/CrossCompilingRaxFile.md). The specs
use `"thread_pool": true`, so no riscv64 libomp is needed.

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/yolov8/specs/k3_pose_320.json \
  --build-dir build \
  --is-rvv-crosscompile \
  --riscv-gnu-toolchain ${BUILD_RISCV_GNU_TOOLCHAIN_DIR} \
  --riscv-mlir-c-runner-utils ${RISCV_MLIR_C_RUNNER_UTILS} \
  --riscv-llvm-build-dir ${RISCV_LLVM_BUILD_DIR} \
  --buddy-mlir-build-dir ${BUDDY_MLIR_BUILD_DIR} \
  --cmake-args=-DBUDDY_RISCV_VLEN=1024 \
  --cmake-args=-DBUDDY_RISCV_ENABLE_ZFH_ZVFH=ON \
  --cmake-args=-DBUDDY_YOLOV8_CHECKPOINT=/path/to/yolov8n-pose.pt
```

Without `BUDDY_YOLOV8_CHECKPOINT`, Ultralytics downloads the checkpoint the
spec names. The package is `build/models/yolov8/yolov8.rax`; the riscv64
`buddy-cli` comes from `make riscv-cli`. VLEN 1024 and Zfh/Zvfh are those of
the K3 A100 cores.

## Lowering

`codegen/lower_to_obj.sh` (the `LOWER_SCRIPT` of `CMakeLists.txt`) lowers the
f32 TOSA graph and runs the parallel loops on the thread pool of the spec (any
RVV target). The linalg ops go through VIR (`-lower-linalg-to-vir`; the
convolutions vectorized along the output channels, the output rows in
parallel), the vectors sized by `-lower-vir-to-vector` for the target of the
llc options (on the K3, `rvv-vlen=1024`).

## Run

Copy `yolov8.rax` and `buddy-cli` to the board and run as an AI process so
the threads run on the A100 cores:

```bash
sh -c 'echo 0 > /proc/set_ai_thread && exec ./buddy-cli --model yolov8.rax --image bus.jpg'
```

```
[0] class_id=0 label=person score=0.8468 box=(45.2, 389.6, 237.1, 905.3)
    keypoints=[(137.4, 439.1, 0.93), (144.8, 429.4, 0.81), ...]
...
Detections: 3
```

Keypoints are `(x, y, visibility)` in image pixels, in COCO order (nose, eyes,
ears, shoulders, elbows, wrists, hips, knees, ankles). `--no-stats` drops the
timings on stderr; `BUDDY_YOLOV8_RUNS=N` runs the forward pass N times and
reports the warm runs.

## Accuracy and speed

SpacemiT K3, 8 A100 cores, yolov8n-pose 320x320, model forward pass (mean of
the warm runs of `BUDDY_YOLOV8_RUNS=20`):

| Build | ms per frame |
| --- | --- |
| default `single_forward` pipeline | 22500 (wrong output) |
| `lower_to_obj.sh`, through VIR | 240 |

The head output of `lower_to_obj.sh` differs from PyTorch by at most 8e-4
pixels (`bus.jpg`, `zidane.jpg`).

The rest of the time per image is outside the model: decoding the JPEG and
letterboxing (`preprocess`, single-threaded, ~75 ms for the 810x1080
`bus.jpg`).
