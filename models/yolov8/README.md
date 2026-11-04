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
the K3 A100 cores; the `"ime"` specs need exactly VLEN 1024.

## The convolutions on the matrix engine (`"ime": true`)

With `"ime"` the importer rewrites the graph (`codegen/yolov8_ime.py`) so that
the model runs on the IME of the A100 cores: every convolution of the
backbone, neck and head (with its SiLU, and the residual add of a bottleneck)
becomes a call of a kernel generated as MLIR for it, and the SPPF max pools,
the upsamplings, the concatenations and the head decode become kernels too.
`codegen/lower_to_obj.sh` lowers them with the IME dialect and links them
with the rest of the graph. The convolutions multiply fp16 operands into f32
accumulators (`smt.vfwmadot`, 8 x 8 x 8 per instruction): **the numerics
change** from f32 (see below).

- Between the kernels an activation is fp16 in a channel-blocked, zero-padded
  layout, `[C / 8][H + 2 + slack][W + 2][8]`: 8 pixels times 8 channels are
  one contiguous operand, the halo is the padding of the 3 x 3 convolutions,
  and a channel split is a view.
- A tile is 2 x 4 blocks of 8 pixels x 8 output channels (16 accumulator
  registers). The operands of the next K step load while the current one
  multiplies; the weights are packed in the order a tile reads them, in fp16
  in the parameter pack, and the most re-read ones are copied into the TCM of
  each core pair at the first call (`runtime/spacemit/BuddySpacemitTcm.c`;
  `BUDDY_SPACEMIT_TCM=0` reads them from memory, with the same results).
- On the A100 an indexed or strided vector access costs 10 to 40 times a
  contiguous one, so every kernel loads and stores contiguously: the input of
  a stride-2 convolution is split by column parity first, the first
  convolution reads a space-to-depth of the image (a 2 x 2 convolution on 12
  channels), the head convolutions compute transposed tiles so that their
  NCHW f32 stores write 8 contiguous pixels per channel.
- The epilogue (bias in the initial accumulators, SiLU with a polynomial
  exp and `vfrec7` reciprocal, residual) interleaves two accumulators for the
  in-order core.

Without `"ime"`, `codegen/lower_to_obj.sh` lowers the f32 TOSA graph and
runs the parallel loops on the thread pool of the spec (any RVV target). The
linalg ops go through VIR (`-lower-linalg-to-vir`; the convolutions
vectorized along the output channels, the output rows in parallel), the
vectors sized by `-lower-vir-to-vector` for the target of the llc options (on
the K3, `rvv-vlen=1024`).

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
the warm runs of `BUDDY_YOLOV8_RUNS`, 20 for f32 and 300 for `"ime"`):

| Build | ms per frame | frames/s |
| --- | --- | --- |
| f32, default `single_forward` pipeline | 22500 (wrong output) | |
| f32, `lower_to_obj.sh`, through VIR | 240 | ~4.2 |
| `"ime"` | 5.1-5.3 | ~190 |

yolov8n (detect) with `"ime"`: 5.1 ms.

Accuracy of `"ime"` against Ultralytics (PyTorch f32, same letterbox, conf
0.25, IoU 0.7) on the 128 images of COCO128: the same 127 person detections,
all matched (IoU > 0.5); score differences mean 0.0003, max 0.005; boxes mean
0.02, max 0.4 pixels; keypoints mean 0.07, p99 0.54, max 2.0 pixels (of the
320 x 320 input). `bus.jpg` and `zidane.jpg`: the same 5 people, scores within
0.005. The f32 build differs from PyTorch by at most 8e-4 pixels in the raw
head output (`bus.jpg`, `zidane.jpg`).

The rest of the time per image is outside the model: decoding the JPEG and
letterboxing (`preprocess`, single-threaded, ~75 ms for the 810x1080
`bus.jpg`).
