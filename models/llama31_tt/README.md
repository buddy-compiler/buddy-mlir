# Llama 3.1 Tenstorrent buddy-cli Path

This directory wires prebuilt Llama 3.1 TTNN flatbuffers into the generic
`buddy-cli --model <file.rax>` entrypoint.

The `.rax` package follows the same payload-embedding path used by the
config-driven buddy-cli models: it embeds the TTNN flatbuffers and chat
artifacts, including the shared Llama weight archive. `buddy-cli` dispatches
the package to the native C++ Tenstorrent runtime in-process.

## Build

```bash
cmake -S . -B build \
  -DBUDDY_BUILD_LLAMA31_TT_MODEL=ON \
  -DBUDDY_ENABLE_TENSTORRENT=ON \
  -DBUDDY_TT_MLIR_BUILD_DIR=$PWD/build-ttmlir
cmake --build build --target buddy-cli llama31_tt_rax
```

The `llama31_tt_rax` target captures the graph, lowers TTIR to TTNN, prepares
`chat_artifacts`, and writes:

```text
build/models/llama31_tt/llama31_tt.rax
```

See [TenstorrentEnvironment.md](../../docs/TenstorrentEnvironment.md).

The full capture, lower, package, and run wrapper is
[`run_llama31_p150_chat.sh`](run_llama31_p150_chat.sh).

## Run With buddy-cli

```bash
ulimit -v 95000000
ulimit -m 95000000
export BUDDY_RAX_PAYLOAD_DIR=/tmp/$USER/buddy_rax_payload
mkdir -p "$BUDDY_RAX_PAYLOAD_DIR"

LLAMA31_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
build/bin/buddy-cli \
  --model build/models/llama31_tt/llama31_tt.rax \
  --prompt "Hello" \
  --max-tokens 32
```

`LLAMA31_MODEL_PATH` must point at a local Llama-3.1-8B-Instruct checkout that
contains the tokenizer files. The native C++ runtime does not download models
or import Python packages.

## Package Existing TTNN Artifacts

The supported path is the `llama31_tt_rax` CMake target above. Use the lower
level manifest command only when both TTNN flatbuffers and `chat_artifacts`
have already been generated.

```bash
cd /path/to/buddy-mlir
export BUDDY_REPO_ROOT=$(pwd)
export BUDDY_BUILD=${BUDDY_BUILD:-$BUDDY_REPO_ROOT/build}

python3 "$BUDDY_REPO_ROOT/tools/buddy-codegen/gen_tenstorrent_manifest.py" \
  --prefill-ttnn "$BUDDY_BUILD/models/llama31_tt/ttir_out_static/llama31_prefill_static_argattrs.ttnn" \
  --decode-ttnn "$BUDDY_BUILD/models/llama31_tt/ttir_out_static/llama31_decode_static_argattrs.ttnn" \
  --artifacts "$BUDDY_BUILD/models/llama31_tt/chat_artifacts" \
  --max-cache-len 1024 \
  -o "$BUDDY_BUILD/models/llama31_tt/llama31_tt.rhal.mlir"

"$BUDDY_BUILD/bin/rax-pack" "$BUDDY_BUILD/models/llama31_tt/llama31_tt.rhal.mlir" \
  --embed-payload \
  -o "$BUDDY_BUILD/models/llama31_tt/llama31_tt.rax"
```

The `.rax` manifest uses:

- `model_name = "llama31_tt"` for buddy-cli dispatch.
- `prefill_ttnn` / `decode_ttnn` code objects with `backend = "ttnn"`.
- `artifact_prefill_*` / `artifact_decode_*` external constants for embedded
  chat artifacts. The prefill and decode weight constants point at the same
  archive so the self-contained `.rax` embeds the weights once.
- `artifacts_uri`, `tokenizer_uri`, and `max_cache_len` module attributes for
  the native Tenstorrent runtime bridge.
