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
sampling options. Every default is false, zero, or empty. DeepSeek enables its
supported features after loading and validates capabilities before HTTP execution.
TT supports raw encoding/decoding with a tiktoken-format `tokenizer.model`
and greedy generation when a compatible serving backend is installed. TT
declares no sampling features, including when its outputs contain logits.
Qwen does not advertise compatible serving features.
See the [DeepSeek serving guide](deepseek.md) and [TT serving guide](llama31-tt.md).

`DetokenizeRequest` carries token IDs and `DetokenizeResult` carries text. The
raw implementation must validate every ID, join raw bytes including
special tokens and bytes after EOS, then repair UTF-8 once. DeepSeek and TT support
`/detokenize`; TT requires `tokenizer.model`. TT's legacy `tokenizer.json` path retains
its historical behavior and does not advertise compatible raw codec support.
Other adapters explicitly reject unsupported decoding; unloaded instances
reject execution.

| DTO addition | Compatible serving meaning |
| --- | --- |
| `SamplingParams.maxNewTokens` | 0: prefill only; -1: context-limited; positive: sampled new-token budget. |
| `explicitParameters` | Records non-null fields after alias selection and normalization, for capability checks. |
| `seedPolicy` / `fixedSeed` | Random or a fixed seed, including zero; the upstream UINT32_MAX sentinel selects Random. |
| `TokenizeRequest.parseSpecial` / `withPieces` | Special-token recognition and optional token pieces. |
| `TokenizeResult.pieces` | Optional pieces accompanying token IDs. |

Compatible serving uses `maxNewTokens` independently of the legacy
`maxTokens` field. Adapters that do not declare generated-token budget support
must reject compatible generation requests. `TokenizeRequest.addSpecial`
retains its historical C++ true default; the raw HTTP tokenizer default is false.
The interface layout does not change numerical algorithms, precision, weights,
budgets, or seeds.

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

The no-weight production source checks use discovery and the real v2
loader, query unloaded capabilities, verify decoding rejection, and repeat
creation/destruction. Existing HTTP tests continue to exercise resident,
embedding, masked-LM, and transcription services. The CLI golden test continues
to execute its historical generation overload.

`buddy-server-llama31-tt-execution`, `buddy-server-llama31-tt-batch-manifest`,
and `buddy-server-llama31-tt-serving` run in the default server test suite even
when model build options are OFF. The serving test injects selected token IDs
into the production execution/resident classes and shared generation loop.
It covers the compatible greedy contract and tokenizer/template behavior;
its programs and vocabulary are synthetic and it does not exercise the TTNN SDK.

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
