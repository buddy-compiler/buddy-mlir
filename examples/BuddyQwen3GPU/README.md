# Buddy Compiler Qwen3-0.6B GPU Example

## Introduction

This example demonstrates running Qwen3-0.6B inference on GPU using Buddy Compiler. The model is compiled from MLIR through a GPU lowering pipeline (TOSA → Linalg → Parallel Loops → GPU → NVVM → LLVM), and executed via the MLIR CUDA runtime.

Key features:
- Single-layer prefill + decode kernel compiled to GPU via MLIR
- Grouped Query Attention (GQA) with 16 Q heads / 8 KV heads
- BPE tokenizer (ByteLevel) embedded in the C++ binary — no Python dependency at runtime
- Qwen3 chat template (`<|im_start|>user ... <|im_end|>`) applied automatically
- Per-token timing and tokens/s statistics

## Prerequisites

### 1. Build LLVM/MLIR with CUDA support

```bash
cd buddy-mlir
mkdir llvm/build && cd llvm/build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
ninja
```

### 2. Build buddy-mlir

```bash
cd buddy-mlir
mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
ninja
```

### 3. Download Qwen3-0.6B model

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /path/to/Qwen3-0.6B
```

## Build

### Option A: CMake (recommended)

```bash
cd buddy-mlir/build
# Change sm_80 to your GPU arch, e.g. sm_86, sm_89
cmake .. -DBUDDY_QWEN3_GPU_EXAMPLES=ON \
         -DCUDA_SM=sm_80
make buddy-qwen3-nextgpu-run -j$(nproc)
```

The binary will be at `build/examples/BuddyQwen3GPU/buddy-qwen3-nextgpu-run`.

### Option B: Makefile (standalone)

```bash
cd buddy-mlir/examples/BuddyQwen3GPU
make next-qwen3-e2e-run -j$(nproc)
```

The binary will be at `examples/BuddyQwen3GPU/buddy-qwen3-nextgpu-run`.

To override the GPU architecture (default: `sm_80`):


## Prepare Weights

Run the export script once to convert the HuggingFace model weights into the binary format expected by the runner:

```bash
cd build/examples/BuddyQwen3GPU    # or the source dir for Makefile builds

python3 import-qwen3-nextgpu.py \
    --model-path /path/to/Qwen3-0.6B \
    --output qwen3_nextgpu_weights.bin
```

Or set the environment variable instead of `--model-path`:

```bash
export Qwen3_0_6B_MODEL_PATH=/path/to/Qwen3-0.6B
```

## Run

```bash
cd build/examples/BuddyQwen3GPU
./buddy-qwen3-nextgpu-run
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-new N` | `200` | Maximum number of tokens to generate |

Example:

```bash
./buddy-qwen3-nextgpu-run --max-new 50
```

## File Structure

```
BuddyQwen3GPU/
├── next-qwen3-prefill-kernel.mlir   # MLIR source for prefill (all layers, full sequence)
├── next-qwen3-decode-kernel.mlir    # MLIR source for decode (all layers, single token)
├── buddy-qwen3-nextgpu-main.cpp     # C++ runner with embedded BPE tokenizer
├── import-qwen3-nextgpu.py          # Weight export script
├── CMakeLists.txt                   # CMake build
└── makefile                         # Standalone make build
```

## Notes

- The KV cache sequence length is fixed at **512** and baked into both the compiled MLIR kernels and the weight export script. The model supports up to 512 tokens of context (prompt + generated tokens combined).
