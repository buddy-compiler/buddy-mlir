# Resident plugin ABI v2

Resident ABI v2 changes the C++ interface and its DTO layout. Build
`buddy-server`, Buddy runtime, and resident plugins from the same sources and
with compatible C++ toolchains. Repackage existing `.rax` files with the rebuilt
resident plugin. Weights, kernels, vocabularies, and chat templates do not need
regeneration for this ABI migration.

The loader requires these three exports:

```cpp
extern "C" buddy::runtime::ResidentModel *buddy_create_resident_model_v2();
extern "C" void buddy_destroy_resident_model_v2(buddy::runtime::ResidentModel *);
extern "C" const char *buddy_resident_model_type_v2();
```

The type export must return a nonempty model-type string. The loader resolves
all three symbols before calling any resident entry point. Missing symbols fail
with the symbol name and instructions to rebuild and repackage. A v1 plugin is
rejected before factory invocation or access to its old vtable; no v1 fallback
is available. Renaming exports on an old binary is insufficient: its interface
and DTO layouts must be rebuilt.

`buddy_model_plugin_descriptor_v1` still identifies the service kind. Its
version does not identify the resident ABI. CLI `runner_library`, embedding,
masked-LM, and transcription factory ABIs retain their existing versions.

## Interface and DTO changes

`ResidentModel` adds two required virtual methods:

```cpp
virtual ResidentModelCapabilities capabilities() const = 0;
virtual DetokenizeResult detokenize(const DetokenizeRequest &) = 0;
```

Capabilities describe features certified by the loaded artifact/configuration:
raw encoding/decoding, tokenizer modes, chat roles and multiple turns, image
count and input forms, streaming, generated-token budgets, stops, and individual
sampling options. Every default is false, zero, or empty. The supplied
production adapters retain these defaults after loading. Their legacy methods
do not use capability declarations for HTTP validation.

`DetokenizeRequest` carries token IDs and `DetokenizeResult` carries text. A
compatible raw implementation must validate every ID, join raw bytes including
special tokens and bytes after EOS, then repair UTF-8 once. All three production
adapters reject raw decoding as unsupported. The HTTP router does not expose
the `/detokenize` route.

| DTO addition | Compatible serving meaning |
| --- | --- |
| `SamplingParams.maxNewTokens` | 0: prefill only; -1: context-limited; positive: sampled new-token budget. |
| `explicitParameters` | Records non-null fields after alias selection and normalization, for capability checks. |
| `seedPolicy` / `fixedSeed` | Random or a fixed seed, including zero; the upstream UINT32_MAX sentinel selects Random. |
| `TokenizeRequest.parseSpecial` / `withPieces` | Special-token recognition and optional token pieces. |
| `TokenizeResult.pieces` | Optional pieces accompanying token IDs. |

`TokenizeRequest.addSpecial` defaults to true for existing C++ callers.
Its raw HTTP contract default is false. HTTP parsing and the supplied
legacy adapters use `maxTokens`; those paths do not consume the
compatible serving fields in the table. Numerical algorithms, precision,
weights, budgets, and seed behavior are unchanged by the interface layout.

## Validation

```bash
ninja -C build -j1 check-buddy-resident-adapters
ninja -C build -j1 check-buddy-server
ninja -C build -j1 check-buddy
```

`buddy-server-resident-plugin-handle` exercises v2 creation, DTO/vtable dispatch,
and matching destruction. It rejects an independent v1 fixture, each missing
required export, null factories, and empty/null model types. Callback markers
prove no resident entry point is invoked before all exports are validated. The
v1 fixture has no dependency on the new `ResidentModel` definition and aborts
if an old entry point is called.

The three no-weight production source checks use discovery and the real v2
loader, query unloaded capabilities, verify unsupported decoding, and repeat
creation/destruction. Existing HTTP tests continue to exercise resident,
embedding, masked-LM, and transcription services. The CLI golden test continues
to execute its historical generation overload.

DeepSeek and Qwen production plugin targets can also be built in separately
configured model build directories without importing weights:

```bash
cmake --build <deepseek-build> --target buddy_models_deepseek_r1_server -- -j1
cmake --build <qwen-build> --target buddy_models_qwen3_vl_server -- -j1
```

The TT portable source check uses `BUDDY_LLAMA31_TT_HAS_TTNN=0`. A separate build
of `buddy_models_llama31_tt_server_runtime` and `buddy_models_llama31_tt_server`
with the real SDK and `BUDDY_LLAMA31_TT_HAS_TTNN=1` is still required for SDK
acceptance. Portable success cannot establish that result. Real model inference,
device checks, and external serving integration remain separate acceptance
steps; an ABI load check does not establish them.
