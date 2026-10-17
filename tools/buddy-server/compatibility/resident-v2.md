# Resident plugin ABI v2

The resident C++ interface and DTO layouts require `buddy-server`, Buddy
runtime, and resident plugins built from the same sources with compatible C++
toolchains. Repackage `.rax` files when rebuilding their resident plugins.
An ABI-only rebuild does not require regenerating weights, kernels,
vocabularies, or chat templates.

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

`buddy_model_plugin_descriptor_v1` identifies the service kind. Its
version does not identify the resident ABI. CLI `runner_library`, embedding,
masked-LM, and transcription factory ABIs retain their existing versions.

## Interface and serving DTOs

`ResidentModel` requires these virtual methods for capability discovery and raw
decoding:

```cpp
virtual ResidentModelCapabilities capabilities() const = 0;
virtual DetokenizeResult detokenize(const DetokenizeRequest &) = 0;
```

Capabilities describe features certified by the loaded artifact/configuration:
raw encoding/decoding, tokenizer modes, chat roles and multiple turns, image
count and input forms, streaming, generated-token budgets, stops, and individual
sampling options. Every default is false, zero, or empty. Adapters declare
supported features after loading, and the server validates capabilities before
HTTP execution. DeepSeek supports raw codecs and sampling. TT supports raw
encoding/decoding with a tiktoken-format `tokenizer.model` and greedy generation
when a compatible serving backend is installed. It declares no sampling
features, including when its outputs contain logits. Qwen supports raw codecs,
greedy streaming generation, supported roles and multiple-turn rendering, and
one local image. Its static package resources constrain executable prompt
lengths and image positions.
See the [DeepSeek serving guide](deepseek.md), [TT serving guide](llama31-tt.md),
and [Qwen serving guide](qwen3-vl.md).

`DetokenizeRequest` carries token IDs and `DetokenizeResult` carries text. The
raw implementation must validate every ID, join raw bytes including
special tokens and bytes after EOS, then repair UTF-8 once. The `/detokenize`
route uses the DeepSeek raw codec, TT's `tokenizer.model`, or Qwen's line-based
byte vocabulary. TT's legacy `tokenizer.json` path does not advertise compatible
raw codec support. Unloaded instances reject execution.

| Serving DTO field | Meaning |
| --- | --- |
| `SamplingParams.maxNewTokens` | 0: prefill only; -1: context-limited; positive: sampled new-token budget. |
| `explicitParameters` | Records non-null fields after alias selection and normalization, for capability checks. |
| `seedPolicy` / `fixedSeed` | Random or a fixed seed, including zero; HTTP seed sentinels select Random. |
| `TokenizeRequest.parseSpecial` / `withPieces` | Special-token recognition and optional token pieces; defaults true/false. |
| `TokenizeResult.pieces` | Optional pieces accompanying token IDs when requested. |

HTTP parsing and serving adapters consume these DTO fields. TT and Qwen use
`ServingTokenSource` with the shared controller's budget/stop semantics and
reject requested sampling features. `TokenizeRequest.addSpecial` defaults to
true for direct C++ callers; the HTTP parser selects false.

`ChatMessage.imagesAfterText` is a false-by-default flag that
preserves whether an image part followed nonempty text in the original content
array. Qwen accepts only image-before-text layouts and rejects this flag.
Direct C++ callers use prefix placement unless they set it. This flag is part of
the v2 DTO layout and requires matching server, runtime, and resident-plugin
builds. Discovery and other service/runner ABIs have independent versions.

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
creation/destruction. HTTP tests exercise resident, embedding, masked-LM, and
transcription services. The CLI golden test executes the legacy generation
overload.

`buddy-server-llama31-tt-execution`, `buddy-server-llama31-tt-batch-manifest`,
and `buddy-server-llama31-tt-serving` run in the default server test suite even
when model build options are OFF. The serving test injects selected token IDs
into the production execution/resident classes and shared generation loop.
It covers the compatible greedy contract and tokenizer/template behavior;
its programs and vocabulary are synthetic and it does not exercise the TTNN SDK.

`buddy-server-qwen3-vl-raw-codec` exercises the repository vocabulary and known
canonical differences. `buddy-server-qwen3-vl-serving` loads synthetic RAX
resources through the production runtime and calls actual `dlopen` kernel
fixtures. It covers serving and image-layout behavior without model arithmetic
or weights. Both run in `check-buddy-server` and `check-buddy`; the narrower
`check-buddy-resident-adapters` target runs plugin link/load checks only.

DeepSeek and Qwen production plugin targets can also be built in separately
configured model build directories without importing weights:

```bash
cmake --build <deepseek-build> --target buddy_models_deepseek_r1_server -- -j1
cmake --build <qwen-build> --target buddy_models_qwen3_vl_server -- -j1
```

The TT portable source check uses `BUDDY_LLAMA31_TT_HAS_TTNN=0`. Compiling the
SDK branch requires `buddy_models_llama31_tt_server_runtime` and
`buddy_models_llama31_tt_server` with the real SDK and
`BUDDY_LLAMA31_TT_HAS_TTNN=1`. Portable link/load checks do not compile that
branch or exercise model inference, devices, or external serving integrations.
