# Llama 3.1 Tenstorrent resident serving

The batch-1 TT resident adapter uses the shared `runServingGeneration`
controller through `ServingTokenSource` for generated-token budgets, stops,
usage, UTF-8 streaming, and cancellation. It supports greedy generation only.
The legacy CLI generation path retains its existing behavior; this serving
change does not alter weights, operators, precision, or sampling formulas.

## Package and tokenizer requirements

Build the canonical `llama31_tt.rax` package with its matching resident v2
plugin, TTNN prefill/decode programs, artifacts, and tokenizer resources. The
server rejects fixed-batch `llama31_tt_bN.rax` packages. Both external manifest
resources and embedded payloads are supported. See the
[model build and runtime instructions](../../../models/llama31_tt/README.md)
and [resident ABI migration guide](resident-v2.md).

Compatible raw encoding/decoding requires a tiktoken-format `tokenizer.model`
containing base64 byte pieces and integer ranks. Loading prefers
`original/tokenizer.model`, then `tokenizer.model`. The historical
`tokenizer.json` fallback remains available to the legacy path, but it does not
declare compatible raw codec or generation capabilities.

`/tokenize` defaults to `add_special=false`, `parse_special=true`, and
`with_pieces=false`. Raw encoding adds no chat wrapper, preserves whitespace,
and does not apply the model context limit or truncate input.
`add_special=true` adds the model BOS. `parse_special=true` recognizes fixed
Llama special tokens and special entries in `tokenizer_config.json`'s
`added_tokens_decoder`, choosing the longest matching special string.
Conflicting special-token IDs or spellings are rejected while loading.
`parse_special=false` encodes those strings as ordinary text.
`count_only=true` returns the actual count without token IDs or pieces.
`with_pieces=true` returns a UTF-8-repaired piece per token; fragments repaired
individually need not concatenate to the result of decoding their joined bytes.

Ordinary spans between recognized special tokens use byte BPE over the whole
span. This implementation does not perform canonical Llama/tiktoken regex
pre-tokenization. Its raw-byte preservation and synthetic-vocabulary tests do
not establish canonical token IDs, segmentation, or equivalent model inputs,
including when using a real Llama vocabulary.

`/detokenize` validates every ID, including negative IDs and holes in sparse
vocabularies. It joins all raw bytes before one UTF-8 repair pass, preserving
leading/repeated spaces, newlines, NUL, special-token text, and content after
EOS. Byte fragments that jointly form a UTF-8 character decode together.
Invalid bytes become U+FFFD and an incomplete final prefix becomes one U+FFFD.
Empty input decodes to an empty string.

## Greedy generation, stops, and chat

The TTNN backend declares compatible generation only after validating both
prefill and decode output descriptors as Float32/BFloat16 logits or
Int32/Int64/UInt32 device argmax IDs. Float16 packages retain legacy loading
and generation, but declare compatible generation unavailable. Supported
outputs use greedy selection. No sampling capabilities are declared, even
when a particular program exposes logits. Omit `temperature`, `top_k`,
`top_p`, `min_p`, `repeat_penalty`,
`repeat_last_n`, and `seed` in HTTP requests. An explicit non-null value is
rejected even if it equals the greedy default, such as `temperature:0`.
Null fields use defaults. Direct C++ serving callers are also rejected for
unsupported nondefault sampler settings.

For batch 1, standard prefill/decode logits have shape `[1,S,V]` and device
IDs have shape `[1,S]`. Leading singleton dimensions are accepted; logits
may also be a vector or `[1,V]`. Other layouts that the existing selector
cannot index, such as `[S,V]` logits with `S > 1` or `[1,S,1]` IDs with
`S > 1`, do not advertise compatible generation.

The generated-token budget uses alias priority
`n_predict > max_completion_tokens > max_tokens`. Zero performs prefill without
selecting a token; -1 runs until a stop or the context boundary; positive values
count newly selected tokens. A selected EOS/stop ID counts in completion usage
and takes precedence over budget exhaustion on the same token. Prompt usage
counts the actual encoded prompt, including BOS/template tokens. A prompt
filling the context may run prefill without selection; an oversized prompt is
rejected. The server does not truncate or move the context.

Request and template string/ID stops are combined. Unless the package has
`ignore_eos` enabled, the metadata EOS and Llama `end_of_text` (128001),
`eom_id` (128008), and `eot_id` (128009) are also included. Custom-template stops
remain active with `ignore_eos`. String stops match raw bytes before UTF-8
repair, including across tokens or within a split code point. Matched stop
content is withheld from the response.

Completion adds BOS when the prompt does not already begin with its spelling.
Chat renders its template once and encodes that result without adding another
BOS. The default template preserves system/user/assistant roles and multiple
turns. A custom Buddy JSON template is parsed and cached while loading; its
role map determines supported roles, BOS, and assistant generation prefix.
For `system_in_first_user`, the supported system shape is one leading system
message immediately followed by a user message. Unsupported roles and image
inputs are rejected.

The request's model alias does not switch the single loaded model; results and
stream chunks identify the loaded model. Resident requests are serialized.
Request-scoped KV state, inputs, and outputs are released on success,
cancellation, or exception, allowing subsequent requests to reuse the session.
When a callback returns false, no further callbacks are emitted; cancellation
before the terminal chunk suppresses it.

## Build and validation boundaries

The following production-source tests are built even when model build options
are OFF. They are included in `check-buddy-server` and `check-buddy`:

```bash
ninja -C build -j1 buddy-server-llama31-tt-execution-test \
  buddy-server-llama31-tt-batch-manifest-test buddy-server-llama31-tt-serving-test
ctest --test-dir build --output-on-failure -j1 \
  -R '^buddy-server-llama31-tt-(execution|batch-manifest|serving)$'
```

The serving test injects a scripted token source into the production
execution/resident classes and shared controller. It covers raw bytes and
special tokens, greedy budgets, template rendering, stop handling,
cancellation, and recovery after failures. Its vocabulary and TT programs are
synthetic fixtures. These tests do not run model weights or the TTNN SDK.

`check-buddy-resident-adapters` runs plugin link/load checks, including
`buddy-server-llama31-tt-resident-portable-link-check` with
`BUDDY_LLAMA31_TT_HAS_TTNN=0`. It does not run the TT serving behavior tests.
The legacy fake-execution environment variable does not provide a compatible
serving token source and does not establish TTNN behavior.

Compiling the TTNN branch requires the Tenstorrent development dependencies
and these production targets with
`BUDDY_LLAMA31_TT_HAS_TTNN=1`:

```bash
cmake --build "$BUDDY_BUILD" --target \
  buddy_models_llama31_tt_server_runtime buddy_models_llama31_tt_server -- -j1
```

Portable link/load checks, scripted generation, and the synthetic HTTP contract
exercise different paths from SDK-enabled model execution. Validate the
production targets with the SDK and run the packaged model on a device to
check inference behavior. Qwen's serving limits are described in the
[Qwen guide](qwen3-vl.md).
