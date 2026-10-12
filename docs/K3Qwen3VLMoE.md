# Qwen3-VL-30B-A3B on the SpacemiT K3 (variant `w4g32`, text only)

The text decoder of Qwen3-VL-30B-A3B-Instruct, a mixture-of-experts model (48
layers, 128 experts of which each token uses 8, 3 B of its 30 B parameters
per token), runs on the 8 A100 cores of a SpacemiT K3 with the `w4g32`
variant of [K3DeepSeekR1.md](K3DeepSeekR1.md): int4 weights with one f16
scale per 32 rows, kernels generated per shape by `import_model.py`. Prompts
are text; the vision encoder is not built.

## Build

As for DeepSeek R1 ([K3DeepSeekR1.md](K3DeepSeekR1.md)), with the
`qwen3_vl_moe` spec and a local copy of the HuggingFace checkpoint:

```bash
python3 tools/buddy-codegen/build_model.py \
  --spec models/qwen3_vl_moe/specs/k3_w4g32.json \
  --build-dir build \
  --local-model ${QWEN3_VL_30B_A3B_DIR} \
  --cmake-args=-DBUDDY_RISCV_VLEN=1024 \
  --cmake-args=-DBUDDY_RISCV_ENABLE_ZFH_ZVFH=ON \
  --is-rvv-crosscompile \
  --riscv-gnu-toolchain ${BUILD_RISCV_GNU_TOOLCHAIN_DIR} \
  --riscv-mlir-c-runner-utils ${RISCV_MLIR_C_RUNNER_UTILS} \
  --riscv-llvm-build-dir ${RISCV_LLVM_BUILD_DIR} \
  --buddy-mlir-build-dir ${BUDDY_MLIR_BUILD_DIR}
# -> build/models/qwen3_vl_moe/qwen3_vl_moe.rax
```

The MoE kernel computes HF Qwen3-MoE's routing with the top-k weights
normalized (`norm_topk_prob`, true for Qwen3-VL-30B-A3B); a model without
it is refused. The import loads the model in f32 (about 120 GB of host memory) and
quantizes all its weights (about 5 minutes on a 64-core host). The weight
files take 18 GB: the board needs 32 GB of memory.

The spec (`models/qwen3_vl_moe/specs/k3_w4g32.json`) is the DeepSeek R1 one
plus:

- `"arena_prefault_mb": 1536`: the model library's arena faults in 1.5 GiB
  when it is loaded instead of the default 512 MiB. A 64-token prefill call
  of this model uses 1.3 GiB; with 512 MiB, the first call faulted in the
  rest and took 2.2 instead of 1.4 s.

## Run

```bash
sh -c 'echo 0 > /proc/set_ai_thread && exec buddy-cli --model qwen3_vl_moe.rax --prompt "..."'
```

With `--chat-template`, the prompt gets the Qwen chat scaffold.

## How it works

| Step | Where |
| --- | --- |
| The text decoder of the vision-language checkpoint, loaded as a Qwen3-MoE causal LM: for text, the interleaved MRoPE of three equal positions is the RoPE of that position | `gen_config.py` (`text_decoder_config`), `import_model.py` (`load_qwen3_moe`) |
| The experts of a MoE block traced as one node, `buddy::moe_experts`: HF computes them in a loop over the selected experts, which depends on the data | `frontend/Python/moe.py`, `MoEExpertsOp` |
| q / k / v without bias in one kernel; Qwen3's RMSNorm over each head of q and k computed by the attention kernel | `k3_w4.py` (`rewrite_qkv`, `rewrite_attention`) |
| Each MoE block one kernel: the router stays f32 (as in llama.cpp); the kernel computes the logits, the top 8 and their weights, then the experts on int4 tiles, with the RMSNorm before the block | `k3_w4.py` (`rewrite_moe`, `pack_moe`, `_moe_fn`) |
| Decode: the residual adds after the o projection and the MoE block computed by those kernels | `k3_w4.py` (`fuse_residual`) |

The MoE kernel quantizes its input to int8 per group of 32, as the matmul
kernels do. Decode (one row) spreads the gate / up tiles of the 8 selected
experts over the threads, then the output tiles, each summing its 8 down
tiles. Prefill groups the rows of a chunk by expert and gives each thread a
range of (expert, tile, block of 4 rows) units, so that each selected
expert's weights are read once per chunk; an expert's partial last block is
computed for its rows only. The
dense layers and the attention of prefill run on the matrix engine
(`prefill_ime`), as for DeepSeek R1.

Decode is bound by DRAM: 1.75 GB per token (the attention weights, the 8
experts of each layer, the LM head). The kernels stream at 24 to 27 GB/s.
With `BUDDY_THREAD_POOL_PROFILE` ([ModelThreadPool.md](ModelThreadPool.md)),
7.2 of 80.8 ms per token were gaps between parallel regions, where one thread
computed the graph's glue ops (RMSNorm, residual adds) alone. Moving those
into the kernels' parallel loops (RMSNorm into the MoE kernel, residual adds
into the o and MoE kernels, q's head norm into the attention's work items)
took decode from 12.5 to 13.1 tok/s.

## Results

SpacemiT K3, 8 A100 cores, `buddy-cli`, greedy, 3 alternating runs of each;
llama.cpp-tools-spacemit with SpacemiT's Q4_1 GGUF of the model, on the same
board:

| | prefill, 64 tokens | prefill, 458 | prefill, 900 | decode, short context | decode after 458 | decode after 900 |
| --- | --- | --- | --- | --- | --- | --- |
| buddy-mlir `w4g32` | 1.26 s | 9.21 s | 16.86 s | 13.05 tok/s | 11.95 tok/s | 10.78 tok/s |
| the same before the decode fusions (below) | 1.28 s | 9.27 s | 16.97 s | 12.51 tok/s | 11.50 tok/s | 10.66 tok/s |
| llama.cpp, Q4_1 (`llama-bench` pp / tg128) | 1.29 s | 8.49 s | 18.23 s | 11.78 tok/s | 10.95 tok/s | 10.27 tok/s |

buddy's prompts are 64, 455 and 896 tokens with its Qwen3 tokenizer (3 to 4
fewer than HuggingFace's); llama-bench uses 64, 458 and 900. Decode is 5 to
11% faster than llama.cpp, prefill faster at 64 and 900 tokens and 8% slower
at 458.

The two formats differ: `w4g32` is symmetric int4 with an f16 scale per 32
weights (4.5 bits per weight), Q4_1 adds an offset (5 bits), and the GGUF
keeps the LM head in Q6_K: llama.cpp reads 2.0 GB per decoded token, buddy
1.75 GB. Prefill 458 tokens is 8 chunks of 64 rows (the last one starts at
row 391 and recomputes 57 rows), about 1.17 s per chunk.

Perplexity on wikitext-2 (test set, chunks of 512 tokens, 16 chunks, the
second half of each scored, `buddy-cli --perplexity`): 7.98.

## Tests

- `tests/Python/test_k3_w4_moe.py`: the MoE kernel (routing included) for
  1, 8 and 32 rows against a numpy reference, with the fused RMSNorm, and the
  fused residual add bit-identical to the add after the kernel; the
  attention with the head norms of q and k, decode and prefill, and the
  `prefill_ime` one with its matrix loops emulated.
- `tests/Python/test_k3_w4_kernels.py`: the matmul kernel with the fused
  residual add, bit-identical to the add after it.
- `tests/Python/test_k3_w4_moe_import.py`: a tiny random Qwen3-MoE through
  `import_model.py`: `buddy::moe_experts` gives HF's logits; both graphs get
  the MoE kernels with the RMSNorm (and in decode the residual add) fused and
  the same packed experts, on which the kernels give HF's MoE blocks; a
  routing other than the kernel's (`norm_topk_prob` false) is refused.
- `tests/Python/test_k3_w4_import.py`: the residual adds of decode fused,
  and not a residual computed after the kernel's call.
