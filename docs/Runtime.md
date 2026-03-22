# Runtime

Buddy Runtime is the inference runtime layer of buddy-mlir. It wraps MLIR compilation outputs into dynamically loadable models and runs them through the generic `buddy-cli` tool.

## End-to-end pipeline

```
models/deepseek_r1/import_deepseek_r1.py
        │
        ▼  Stage 1: Python → MLIR (download model from Hugging Face)
forward_prefill.mlir  subgraph0_prefill.mlir
forward_decode.mlir   subgraph0_decode.mlir   arg0.data
        │
        ▼  Stage 2: MLIR → .o (buddy-opt / mlir-opt / llc toolchain)
forward_prefill.o  subgraph_prefill.o
forward_decode.o   subgraph_decode.o
        │
        ▼  Stage 3: Link → .so
deepseek_r1_model.so
        │
        ▼  Stage 4: rax-pack → .rax (FlatBuffer manifest) + auto-copy vocab.txt
deepseek_r1.rax ──────► buddy-cli --model deepseek_r1.rax --prompt "..."
vocab.txt               (runtime: dlopen + inference)
```

All stages are driven by `models/deepseek_r1/CMakeLists.txt` and complete under a single output directory.
After `ninja deepseek_r1_rax`, `build/models/deepseek_r1/` contains everything needed for inference except the `arg0.data` weight file (~7 GB, not shipped in the source tree).

---

## Architecture

### Layered design

```
┌───────────────────────────────────────────┐
│  buddy-cli  (tools/buddy-cli/)            │  ← Generic entry: reads model_name from .rax for dispatch
├───────────────────────────────────────────┤
│  InferenceRunner  (runtime/core/)         │  ← Abstract interface; one subclass per model
├───────────────────────────────────────────┤
│  DeepSeekR1Runner (models/deepseek_r1/)   │  ← Full loop: tokenize → prefill → decode
│  ModelSession     (models/deepseek_r1/)   │  ← dlopen + KV cache (56 layers) + prefill/decode
├───────────────────────────────────────────┤
│  deepseek_r1_model.so                     │  ← dlopen at runtime; buddy-cli has zero compile-time link to model
│  (_mlir_ciface_forward_prefill/decode)    │
└───────────────────────────────────────────┘
```

**C++ namespaces**: Shared runtime APIs (`InferenceRunner`, `ModelManifest`, `BufferPool`, `ModelSession`, etc.) live in the nested namespace **`buddy::runtime`**, alongside the compiler frontend `buddy::` and the RHAL dialect `buddy::rhal`. CMake target names remain `buddy_runtime_core` / `buddy_models_deepseekr1` (build-system identifiers only).

`buddy-cli` reads the `model_name` field from `.rax` and constructs the matching `InferenceRunner` via `makeRunner()`. To add a model:

1. Implement an `InferenceRunner` subclass under `models/<new_model>/`
2. Build its static library and wire it in `models/CMakeLists.txt`
3. Link it from `tools/buddy-cli/CMakeLists.txt`
4. Add one `if` branch in `makeRunner()` in `tools/buddy-cli/buddy-cli.cpp`

### Dynamic loading path

```
buddy-cli --model deepseek_r1.rax
  │
  ├─ ModelManifest::loadFromRax()       Read FlatBuffer → modelName / soPath / weightsPath
  ├─ makeRunner("deepseek_r1_fp32")    Construct DeepSeekR1Runner
  └─ runner->run(cfg)
       ├─ ModelSession::createFromRax()
       │    ├─ allocateKVCache()        56 × tensor<1×2×1024×128×f32> (owned by session)
       │    └─ dlopen(soPath)
       │         ├─ dlsym("_mlir_ciface_forward_prefill")
       │         └─ dlsym("_mlir_ciface_forward_decode")
       ├─ loadWeights(weightsPath)      mmap ~7 GB → MemRef<float,1>
       ├─ tokenize(prompt)
       ├─ session->prefill(weights, tokens)   → first token
       └─ session->decode(weights, tok) × N   → following tokens (loop)
```

The `buddy-cli` binary carries no model symbols (`nm buddy-cli | grep ciface` is empty).

### KV cache shape

Each KV buffer has shape `tensor<1 × 2 × 1024 × 128 × f32>`:

| Dimension | Value | Meaning | CMake variable |
|-----------|-------|---------|----------------|
| batch | 1 | Single-sequence inference | — |
| num_kv_heads | 2 | GQA KV head count | `BUDDY_DSR1_HEAD_NUM` |
| max_seq_len | 1024 | Context window | `BUDDY_DSR1_MAX_TOKEN_LEN` |
| head_dim | 128 | Per-head dimension | `BUDDY_DSR1_HIDDEN_SIZE` |

56 buffers = 28 Transformer decoder layers × 2 (K and V stored separately).
Total KV cache memory: `56 × 2 × 1024 × 128 × 4 bytes = 56 MB`.

### Five buffer roles

| Role | Resource | Lifetime | Ownership |
|------|----------|----------|-----------|
| `Constant` | Model weights `arg0.data` | Module | Loaded on host; borrowed by session |
| `Input` | Prompt tokens / decode token | Call | Host |
| `Output` | Logits | Call | Inside session |
| `State` | kv0..kv55 (56-layer KV cache) | Session | **Runtime (session owns)** |
| `Workspace` | Temp buffers from compiler | Runtime-internal | Runtime |

---

## RAX tools

### rax-pack

Packs RHAL MLIR dialect (`.mlir`) into a binary `.rax`:

```bash
build-runtime/bin/rax-pack models/deepseek_r1/deepseek_r1.mlir -o deepseek_r1.rax
```

The `.mlir` uses standard MLIR syntax with `rhal` dialect ops:

```mlir
rhal.module @deepseek_r1 attributes {
    version = "0.1.0",
    model_name = "deepseek_r1_fp32",
    vocab_uri = "file:vocab.txt"} {
  rhal.constant @weights {id = 1 : i32, storage = "external",
                           type = tensor<1777088064xf32>,
                           uri = "file:arg0.data"}
  rhal.codeobj @model_kernels {id = 1 : i32, kind = "host_shared_lib",
                                backend = "cpu", uri = "file:deepseek_r1_model.so"}
  rhal.buffer @prefill_tokens {space = "host", type = tensor<1x1024xi64>}
  // ...
  rhal.func @forward_prefill {
    inputs   = ["prefill_tokens"],
    outputs  = ["logits_prefill"],
    dispatch = "model_kernels",
    args     = ["prefill_tokens", "kv0", ..., "kv55", "logits_prefill"]}
}
```

### rax-inspect

```bash
build-runtime/bin/rax-inspect <file.rax>
```

---

## FAQ

**Q: I changed `deepseek_r1.mlir` (e.g. weight path). Do I need to rebuild C++?**

No—repack only:

```bash
ninja deepseek_r1_rax
```

Paths live in the `.rax` FlatBuffer; C++ reads them at runtime via `ModelManifest`.

**Q: Do I need to prepare `vocab.txt` manually?**

No. `ninja deepseek_r1_rax` copies `examples/BuddyDeepSeekR1/vocab.txt` (already in the repo) next to the build outputs. Only `arg0.data` (~7 GB) is omitted from the repo; when using Mode A/B, symlink it into the build directory.

**Q: `dlopen` fails with `libomp.so: not found`?**

The `.so` is linked with `-Wl,-rpath,${LLVM_LIBRARY_DIR}`. If you still see this, the `.so` may be stale—rebuild:

```bash
ninja deepseek_r1_model_so
```

**Q: How do I set OpenMP compile thread count?**

```bash
cmake -DDEEPSEEKR1_NUM_THREADS=96 ...
```

This is a **compile-time** knob for parallelism in the generated MLIR pipeline. Runtime thread count is controlled by `OMP_NUM_THREADS`.

**Q: After changing the `rax.fbs` schema, how do I refresh `RAX.h`?**

`RAX.h` is generated by `flatc` from `rax.fbs` under `build/runtime/include/buddy/runtime/rax/` and renamed by CMake (it does not live in the source tree). Ninja regenerates it when `rax.fbs` changes. Code should `#include "buddy/runtime/rax/RAX.h"`.
