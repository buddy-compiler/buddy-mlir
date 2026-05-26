# Buddy Compiler DeepSeekR1 GPU Example

## Introduction

This example runs DeepSeekR1-Distill-Qwen-1.5B inference on NVIDIA GPUs using Buddy Compiler. The model is compiled through MLIR's GPU pipeline (parallel-loops -> NVVM) with CUDA managed memory for seamless CPU/GPU data sharing.

## Prerequisites

- NVIDIA GPU with compute capability >= 8.0 (A100, A800, H100, etc.)
- CUDA toolkit installed (nvcc, libcuda, libcudart)
- Python environment with PyTorch and Transformers

```bash
conda activate <your-env>
pip install -r requirements.txt
```

## Build

### 1. Build LLVM/MLIR with CUDA support

```bash
cd buddy-mlir/llvm/build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLLVM_ENABLE_ASSERTIONS=ON
ninja check-mlir
```

### 2. Build Buddy Compiler

```bash
cd buddy-mlir/build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_ENABLE_OPENCV=OFF
ninja buddy-opt
```

### 3. Set environment variables

```bash
export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
export DEEPSEEKR1_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B
```

### 4. Build and run

```bash
# Target a specific GPU architecture (default: sm_80 for A100)
cmake -DCUDA_SM=sm_80 ..
ninja buddy-deepseek-r1-gpu-run
```

The build will:
1. Run the import script to generate MLIR files and parameter data from the HuggingFace model
2. Compile `forward_prefill.mlir` and `forward_decode.mlir` (CPU orchestration with GPU memory management)
3. Compile `subgraph0_prefill.mlir` and `subgraph0_decode.mlir` (GPU compute kernels via NVVM pipeline)
4. Link everything into `buddy-deepseek-r1-gpu-run`

### 5. Run inference

```bash
cd buddy-mlir/build

# Select GPU device
export CUDA_VISIBLE_DEVICES=0

# Run with library path
LD_LIBRARY_PATH=/path/to/conda/env/lib:$LD_LIBRARY_PATH \
    ./bin/buddy-deepseek-r1-gpu-run
```

The program will prompt for input text. Example:

```
DeepSeekR1 GPU Inference Powered by Buddy Compiler

Please send a message:
>>> Hello

[Iteration 0] Token: Alright | Time: 235.38s
[Iteration 1] Token: , | Time: 1.49s
[Iteration 2] Token: the | Time: 1.45s
...
```

## Architecture

```
main.cpp
  |
  |-- forward_prefill (CPU, standard LLVM pipeline + GPU memory mgmt)
  |     |-- subgraph0_prefill (GPU, NVVM pipeline)
  |
  |-- forward_decode (CPU, standard LLVM pipeline + GPU memory mgmt)
        |-- subgraph0_decode (GPU, NVVM pipeline)
```

- **Forward functions**: Orchestrate the inference loop on CPU. Slice model parameters, manage KV caches, call GPU subgraphs. Compiled with `convert-memcpy-to-gpu` pass + `gpu-to-llvm` lowering to allocate buffers as CUDA managed memory.
- **Subgraph functions**: Run the actual compute on GPU. Linalg ops are lowered to parallel loops, mapped to GPU blocks, outlined as CUDA kernels, and compiled to CUBIN via the NVVM pipeline.
- **CUDA managed memory**: All GPU allocations use `cuMemAllocManaged`, making buffers accessible from both CPU and GPU without explicit transfers. This simplifies KV cache passing between prefill and decode phases.

## Performance

Measured on NVIDIA A100 80GB PCIe:

| Phase | Performance |
|-------|-------------|
| Prefill (1024 tokens) | ~235s |
| Decode | ~1.4s/token |

Note: Current performance is unoptimized (1 thread per GPU block, no tiling, no shared memory). Significant speedups are possible with kernel optimization.
