# Llama 3.1 Tenstorrent buddy-cli Path

This directory wires prebuilt Llama 3.1 TTNN flatbuffers into the generic
`buddy-cli --model <file.rax>` entrypoint.

The current integration is intentionally thin: the `.rax` manifest describes
the Tenstorrent artifacts, and `buddy-cli` dispatches to the Python
`ttrt.runtime` runner in this directory. That gives RuyiAI Serving a
stable buddy-cli contract first while keeping tt-metal runtime linkage optional.

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
  -o build/models/llama31_tt/llama31_tt.rax
```

Then run:

```bash
BUDDY_TT_PYTHON=python3 \
build/bin/buddy-cli \
  --model build/models/llama31_tt/llama31_tt.rax \
  --prompt "Hello" \
  --max-tokens 32
```

The `.rax` manifest uses:

- `model_name = "llama31_tt"` for buddy-cli dispatch.
- `prefill_ttnn` / `decode_ttnn` code objects with `backend = "ttnn"`.
- `runner_uri`, `artifacts_uri`, `tokenizer_uri`, and `max_cache_len` module
  attributes for the Python Tenstorrent runtime bridge.
- Optional `device_token_loop` and `ignore_eos` attributes for perf-only decode
  graphs that emit device-resident token ids instead of host logits.

Next patches should replace the external Python bridge with a native
Tenstorrent runtime session once the C++ TT runtime dependency boundary is
settled.
