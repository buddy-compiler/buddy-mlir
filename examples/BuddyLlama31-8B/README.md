# Buddy Llama 3.1 8B TTIR Path

This example is the end-to-end Llama 3.1 path for Tenstorrent P150A:

```text
PyTorch/HF model
  -> Buddy Graph
  -> TTIR MLIR
  -> TTNN MLIR
  -> .ttnn flatbuffers
  -> llama31_tt .rax manifest
  -> buddy-cli
```

The precision baseline is Tenstorrent's official
`tt-metal/models/tt_transformers/demo/simple_text_demo.py` Llama 3.1 demo, not
HF CPU logits.

## Main Files

| File | Purpose |
|---|---|
| `_env.sh` | Local environment helper for Buddy + tt-mlir paths |
| `buddy-llama31-lower-ttir.py` | Capture PyTorch graph with Buddy and lower to TTIR |
| `llama31_chat_prepare.py` | Export runtime slots, weights, shapes, and dtypes |
| `llama31_chat_run.py` | Execute prefill/decode `.ttnn` flatbuffers through `ttrt.runtime` |
| `llama31_official_demo_align.py` | Export/replay the official demo token-matching contract |
| `run_llama31_p150_chat.sh` | Full compile/package/run wrapper |

## Environment

First build Buddy with the Tenstorrent runner:

```bash
cmake -S . -B build \
  -DBUDDY_BUILD_LLAMA31_TT_MODEL=ON
cmake --build build --target buddy-cli rax-pack
```

For TTIR -> TTNN compilation and P150A execution, configure the optional
Tenstorrent environment described in
[`docs/TenstorrentEnvironment.md`](../../docs/TenstorrentEnvironment.md).

Then source the local helper:

```bash
source examples/BuddyLlama31-8B/_env.sh
```

Useful overrides:

```bash
export TTMLIR_SOURCE=/path/to/tt-mlir
export TTMLIR_BUILD=/path/to/tt-mlir/build
export BUDDY_BUILD=/path/to/buddy-mlir/build
export LLAMA31_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct
```

## End-to-End Run

```bash
cd examples/BuddyLlama31-8B
MAX_CACHE_LEN=1024 MAX_NEW_TOKENS=32 ./run_llama31_p150_chat.sh
```

By default the wrapper:

1. lowers prefill/decode Buddy graphs to TTIR,
2. compiles TTIR to TTNN flatbuffers,
3. prepares runtime artifacts,
4. packages the `.ttnn` files as `tt_package/llama31_tt.rax`,
5. runs through `buddy-cli --model tt_package/llama31_tt.rax`.

Set these for incremental work:

```bash
SKIP_LOWER=1
SKIP_PREPARE=1
SKIP_PACKAGE=1
RUN_WITH_BUDDY_CLI=0
```

For decode performance experiments, enable the device-side argmax path:

```bash
DEVICE_ARGMAX=1 MAX_NEW_TOKENS=32 ./run_llama31_p150_chat.sh
```

This emits argmax token ids from the decode graph and sets
`device_token_loop=true` in the `.rax` manifest. It is a perf-only mode: the
first generated token is decoded on the host, then later tokens stay resident
on device and are printed as `<device-token>`. The wrapper writes separate
`*_argmax.ttnn`, `chat_artifacts_argmax`, and `tt_package_argmax` outputs. When
normal `chat_artifacts` already exists, the argmax metadata reuses its
`weights.npz` payloads by hard link or symlink instead of copying another
weight package.

## Official Demo Alignment

Do not claim final precision by comparing only with HF CPU logits. Buddy should
match the official demo contract:

- same model variant and tokenizer,
- same prompt/reference tokens,
- same teacher-forcing policy for token matching,
- same generated token stream or an explicitly explained divergence,
- same Top-1/Top-5 metrics for `ci-token-matching`.

After exporting official reference artifacts:

```bash
python llama31_official_demo_align.py export-reference
```

Buddy replay mode:

```bash
python llama31_chat_run.py \
  --prefill-ttnn ttir_out_static/llama31_prefill_static_argattrs.ttnn \
  --decode-ttnn ttir_out_static/llama31_decode_static_argattrs.ttnn \
  --artifacts chat_artifacts \
  --max-cache-len 1024 \
  --ignore-system-desc \
  --official-reference-npz official_demo_artifacts/ci_token_matching_reference.npz \
  --official-trace-out official_demo_artifacts/buddy_ci_token_matching_trace.json
```

Generated artifacts are ignored by this directory's `.gitignore`.
