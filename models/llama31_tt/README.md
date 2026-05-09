# Llama 3.1 Tenstorrent buddy-cli Path

This directory wires prebuilt Llama 3.1 TTNN flatbuffers into the generic
`buddy-cli --model <file.rax>` entrypoint.

The `.rax` package follows the same payload-embedding path used by the
config-driven buddy-cli models: it embeds the TTNN flatbuffers, Python runner,
and chat artifacts, including the shared Llama weight archive. `buddy-cli`
dispatches the package to the Python `ttrt.runtime` runner while keeping
tt-metal runtime linkage optional.

## Build

```bash
cmake -S . -B build \
  -DBUDDY_BUILD_LLAMA31_TT_MODEL=ON
cmake --build build --target buddy-cli rax-pack
```

When compiling TTIR to TTNN or running on P150A, also configure the optional
Tenstorrent environment checks:

```bash
git submodule update --init --depth 1 thirdparty/tt-mlir
source thirdparty/tt-mlir/env/activate

cmake -S . -B build \
  -DBUDDY_BUILD_LLAMA31_TT_MODEL=ON \
  -DBUDDY_ENABLE_TENSTORRENT=ON \
  -DBUDDY_TT_MLIR_BUILD_DIR=$PWD/thirdparty/tt-mlir/build
```

See [TenstorrentEnvironment.md](../../docs/TenstorrentEnvironment.md).

The full capture, lower, package, and run wrapper is
[`run_llama31_p150_chat.sh`](run_llama31_p150_chat.sh).

## Package Existing TTNN Artifacts

```bash
python3 tools/buddy-codegen/gen_tenstorrent_manifest.py \
  --prefill-ttnn models/llama31_tt/ttir_out_static/llama31_prefill_static_argattrs.ttnn \
  --decode-ttnn models/llama31_tt/ttir_out_static/llama31_decode_static_argattrs.ttnn \
  --artifacts models/llama31_tt/chat_artifacts \
  --runner models/llama31_tt/llama31_chat_run.py \
  --max-cache-len 1024 \
  -o build/models/llama31_tt/llama31_tt.rhal.mlir

build/bin/rax-pack build/models/llama31_tt/llama31_tt.rhal.mlir \
  --embed-payload \
  -o build/models/llama31_tt/llama31_tt.rax
```

Then run:

```bash
BUDDY_TT_PYTHON=python3 \
LLAMA31_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
build/bin/buddy-cli \
  --model build/models/llama31_tt/llama31_tt.rax \
  --prompt "Hello" \
  --max-tokens 32
```

If `LLAMA31_MODEL_PATH` is not set, the runner uses the default Hugging Face id
`meta-llama/Llama-3.1-8B-Instruct`. With offline mode enabled, that model and
tokenizer must already be present in the local Hugging Face cache.

The `.rax` manifest uses:

- `model_name = "llama31_tt"` for buddy-cli dispatch.
- `prefill_ttnn` / `decode_ttnn` code objects with `backend = "ttnn"`.
- `runner_py` as a Python `raw_bytes` code object.
- `artifact_prefill_*` / `artifact_decode_*` external constants for embedded
  chat artifacts. The prefill and decode weight constants point at the same
  archive so the self-contained `.rax` embeds the weights once.
- `runner_uri`, `artifacts_uri`, `tokenizer_uri`, and `max_cache_len` module
  attributes for the Python Tenstorrent runtime bridge.
- Optional `device_token_loop` and `ignore_eos` attributes for perf-only decode
  graphs that emit device-resident token ids instead of host logits.

Next patches should replace the external Python bridge with a native
Tenstorrent runtime session once the C++ TT runtime dependency boundary is
settled.
