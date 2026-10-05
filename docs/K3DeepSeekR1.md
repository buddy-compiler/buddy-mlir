# DeepSeek R1 on the SpacemiT K3 (variant `w4g32`)

The `w4g32` variant of DeepSeek-R1-Distill-Qwen-1.5B runs on the 8 A100 cores
of a SpacemiT K3 (riscv64, RVV 1.0 with VLEN 1024). Every Linear layer is an
int4 matrix with one f16 scale per 32 rows, and the layers and the attention
are calls to kernels that `import_model.py` generates per shape as MLIR. It is
cross-compiled on an x86 host into one `.rax` file for `buddy-cli`.

## Build

Prepare the cross-compilation inputs as in
[CrossCompilingRaxFile.md](CrossCompilingRaxFile.md) (a RISC-V GNU toolchain,
the riscv64 OpenMP and `mlir_c_runner_utils` libraries, a cross-compiled LLVM
build), then:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/k3_w4g32.json \
  --build-dir build \
  --target deepseek_r1_rax \
  --cmake-args=-DBUDDY_RISCV_VLEN=1024 \
  --cmake-args=-DBUDDY_RISCV_ENABLE_ZFH_ZVFH=ON \
  --is-rvv-crosscompile \
  --riscv-gnu-toolchain ${BUILD_RISCV_GNU_TOOLCHAIN_DIR} \
  --riscv-mlir-c-runner-utils ${RISCV_MLIR_C_RUNNER_UTILS} \
  --riscv-llvm-build-dir ${RISCV_LLVM_BUILD_DIR} \
  --buddy-mlir-build-dir ${BUDDY_MLIR_BUILD_DIR}
# -> build/models/deepseek_r1/deepseek_r1.rax (1.8 GB, payload embedded)
```

- `BUDDY_RISCV_VLEN=1024` tells llc the exact VLEN (`+zvl1024b`,
  `-riscv-v-vector-bits-max=1024`): a weight tile is then one e8 register and
  a 512-byte load is one `vl4r`. The kernels are correct for any VLEN, only
  slower.
- `BUDDY_RISCV_ENABLE_ZFH_ZVFH=ON`: the weight scales are f16 vectors.

The spec (`models/deepseek_r1/specs/k3_w4g32.json`):

```json
{
  "variant": "w4g32",
  "max_token_len": 1024,
  "num_threads": 8,
  "prefill_chunk": 64,
  "arena": true,
  "hugepages": true,
  "thread_pool": true
}
```

`w4g32` needs `prefill_chunk` ([ChunkedPrefill.md](ChunkedPrefill.md)): the
prefill graph is traced like decode with 64 tokens per call, and both graphs
use the same kernels with 64 rows and one row. `arena` and `hugepages` are the
memory options of [ModelMemoryOptions.md](ModelMemoryOptions.md); with
`thread_pool` ([ModelThreadPool.md](ModelThreadPool.md)) the parallel loops run
on a pinned pool of threads instead of libomp, so `--riscv-omp-shared` is not
needed. The variant
supports Qwen2 models (`Qwen2ForCausalLM`, untied embeddings) whose Linear
layers are `[K, N]` with K a multiple of 32 and N a multiple of 128;
`gen_config.py` checks this and computes the weight buffer sizes from the
HuggingFace config.

## Run

The K3 kernel only lets "AI" threads run on the A100 cores: make the process
one (`/proc/set_ai_thread`) before `buddy-cli` starts, so that all its threads
run there.

```bash
sh -c 'echo 0 > /proc/set_ai_thread && exec buddy-cli --model deepseek_r1.rax --prompt "..."'
```

## How it works

| Piece | Where |
| --- | --- |
| Variant `w4g32`: f32 parameters and int4 tiles in two weight buffers, sized from the HF config | `gen_config.py` (`w4g32_param_counts`) |
| Graph rewrite: every Linear becomes a kernel call; q / k / v share one call, gate / up / SiLU / mul one call; the RMSNorm in front of a kernel is computed by it; RoPE, the KV cache update and attention over the cache become one call that visits the positions up to the current one | `frontend/Python/graph/transform/k3_w4.py` (`k3_w4_rewrite`), `import_model.py` (`apply_k3_w4`) |
| Kernels built per shape with the MLIR Python bindings (scf / vector / memref, `scf.parallel` over the threads), written to `k3_kernels-w4g32.mlir` | `k3_w4.py` (`build_kernels`) |
| Their compilation, linked into the model library | `compile_pipeline.py` (pipeline `kernels`), `buddy_model.cmake` |
| The KV caches of a prefill chunk updated in place (`-eliminate-memref-copy`, as for decode) | `compile_pipeline.py` |

Weight layout: the columns of a weight `[K, N]` are split into tiles of 128
(one e8 register at VLEN 1024), each one contiguous block, group by group:
16 rows of 128 bytes (two int4 rows per byte) and the 128 f16 scales of the
group. The scale of a group is its signed maximum / -8 and the values are in
[-8, 7], as llama.cpp Q4_0 rounds them. A thread streams its tiles linearly.
The activations are quantized to int8 per 32 elements on the fly; the products
of a group accumulate exactly in int16 (`vwmacc`) and are scaled into f32 once
per group. A decoded token reads 868 MB of weights.

The attention kernel reads q / k / v as the `[rows, heads * 128]` outputs of
the q / k / v kernel and writes the `[rows, heads * 128]` input of the o
projection, so that no head views and no copies are left between the kernels.

## Results

SpacemiT K3, 8 A100 cores, `buddy-cli`, greedy:

| | prefill, 458 tokens | decode after 458 tokens | decode after a short prompt |
| --- | --- | --- | --- |
| buddy-mlir `w4g32` | 9.5 s (48 tok/s) | 23.8 tok/s | 26.6 tok/s |
| llama.cpp-tools-spacemit 0.1.9, Q4_0 | 1.96 s (pp458: 234 tok/s) | | 25.0 tok/s (tg128) |

llama.cpp was measured on the same board; `llama-bench` decodes from an empty
context. Decode streams the weights from DRAM (868 MB per token) and is 6%
faster than llama.cpp. Prefill runs the same RVV kernels with 64 rows per call (and
the LM head on the last row of a chunk only); it does not use the matrix
engine of the A100 cores yet.

The kernels also run on other targets, more slowly: on x86 (48 threads of a
Xeon Platinum 8575C) the same build prefills the 458 tokens in 1.7 s and
decodes 68-81 tok/s.

## Tests

- `tests/Python/test_k3_w4_kernels.py`: the kernels, compiled for the host by
  the `kernels` pipeline and run against numpy references (int4 matmuls
  plain / with bias / with RMSNorm / q, k, v / gate, up; one row and several;
  attention for decode and for a prefill chunk, KV cache update included).
- `tests/Python/test_k3_w4_import.py`: a tiny random Qwen2 through the import:
  no Linear or attention op left, the kernels generated, the weight buffers of
  the sizes `gen_config.py` computed.
