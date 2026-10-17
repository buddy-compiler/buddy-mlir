# DeepSeek resident serving

The DeepSeek resident adapter uses the shared serving controller for generated
token budgets, sampling, stops, usage, UTF-8 streaming, and cancellation. The
legacy CLI tokenizer, decoder, and generation overload retain their behavior.
Model weights, operators, precision, and sampling formulas are unchanged.

## Vocabulary and raw text

`DeepSeekR1RawCodec` loads the line-based vocabulary packaged with the artifact.
The current DeepSeek packaging rule copies
`examples/BuddyDeepSeekR1/vocab.txt`. Raw encoding never adds a chat wrapper,
collapses whitespace, or silently truncates input. `/tokenize` defaults to
`add_special=false`, `parse_special=true`, and `with_pieces=false`.

`add_special=true` adds the vocabulary's model BOS token; an existing initial
BOS is retained once. `parse_special=true` recognizes the supported special
vocabulary strings. With that option disabled, the same strings are encoded as
ordinary text. `count_only` counts the actual encoding and omits the token list.
`with_pieces=true` is unsupported and is explicitly rejected.

The vocabulary contains byte-BPE symbols, raw Unicode pieces, and duplicate
pieces. It does not include the original Hugging Face merge ranks or tokenizer
normalization configuration. Encoding therefore uses deterministic
longest-score segmentation of the loaded raw byte pieces, choosing the first
vocabulary ID for duplicate pieces. This is a documented Buddy tokenizer
difference: text preservation does not establish equivalent token IDs,
segmentation, or prompt logits compared with Hugging Face or llama.cpp.

The production-vocabulary regression records nine sample encodings from the
official `DeepSeek-R1-Distill-Qwen-1.5B` tokenizer at revision
`ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`, with added special tokens disabled.
Representative expected IDs are:

| Text (escaped) | Token IDs |
| --- | --- |
| `  Hello\n\tworld` | `220, 21927, 198, 76508` |
| `你好，中\n` | `108386, 3837, 15946, 198` |
| `<think> A` | `151648, 362` |
| `Hello, world!` | `9707, 11, 1879, 0` |

Matching these samples does not establish general tokenizer equivalence.
For example, `antidisestablishmentarianism` encodes to
`15359, 4243, 33400, 76, 79680, 72, 67813` with this codec, versus
`517, 84242, 33400, 478, 8821, 2142` with the official tokenizer. Both represent
the same input text; their token sequences and resulting model inputs differ.
The regression also checks this known difference and raw text round trips,
including special-token text followed by ordinary text after EOS. It does not
measure real-model inference quality or canonical tokenizer accuracy.

Raw decoding validates all IDs, joins every token's bytes in order, and then
repairs UTF-8 once with `Utf8StreamDecoder`. It preserves leading/repeated
spaces, newlines, NUL, special-token text, and content after EOS. Invalid bytes
become U+FFFD; an incomplete trailing UTF-8 prefix becomes one U+FFFD. Token byte
fragments that together form a character are decoded together. Empty input
returns an empty string. Generation uses the same raw pieces with incremental
UTF-8 repair and the controller's stopping rules. Generated logits can have a
padded width larger than the raw vocabulary; serving samples only the decodable
vocabulary prefix, so padding IDs cannot become output or stop tokens.

## Chat and capabilities

Completion adds the model BOS without a chat wrapper. Chat requires a configured
Buddy JSON template, such as `examples/BuddyDeepSeekR1/deepseek-r1.json`. The
adapter renders supported message roles and turns once, then encodes that exact
prompt. The template controls whether chat has a BOS and an assistant generation
prefix. Unsupported roles and image inputs are rejected.

For templates using `system_in_first_user`, the supported system-message shape
is one leading system message immediately followed by a user message. Other
placements are rejected because the existing template renderer cannot preserve
their semantics. The template's role map determines the advertised roles.
Requests may supply an arbitrary model string; responses identify the loaded
artifact or the configured display name.

Before loading, all capabilities are conservative defaults. After successful
session, vocabulary, and template initialization, the adapter advertises raw
encoding/decoding, the implemented special-token modes, streaming, generated
token budgets, string/token-ID stops, and the shared sampler's temperature,
top-k/top-p/min-p, repetition-window/penalty, and seed support. Chat roles and
multiple-turn support require the configured template. Image and token-piece
output capabilities remain disabled.

Serving budgets count sampled new tokens, including a sampled EOS or stop ID.
Zero performs prefill without sampling; -1 uses the remaining context. Prompt
usage counts the actual encoded prompt, including any BOS/template tokens.
Overlong prompts are rejected. The adapter does not truncate or move context.
Explicit seed zero is reproducible; default/random seed policy retains random
seeding. Cancellation and execution errors leave the session reusable.

## Build and validation

The default no-weight checks compile the production adapter, raw codec, and
the real generated `ModelSession`:

```bash
ninja -C build -j1 check-buddy-resident-adapters
ninja -C build -j1 check-buddy-server
```

The DeepSeek serving test injects a scripted `LLMSession` through the production
adapter's session factory. It exercises the same loading, codec, template,
capability, and generation code without replacing the generated session header.
It covers EOS/budget accounting, stop matching, seed zero, cancellation and
failure recovery, role preservation, and templates with BOS disabled. This is
an adapter integration check; it does not execute model kernels or weights.

For the production plugin, configure a model build with the artifact's matching
DeepSeek spec and local HF configuration. For the default f32 variant:

```bash
cmake -G Ninja -S . -B build-deepseek \
  -DMLIR_DIR="$PWD/llvm/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm/build/lib/cmake/llvm" \
  -DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON \
  -DBUDDY_DSR1_HF_CONFIG="$MODEL_METADATA/config.json"
cmake --build build-deepseek \
  --target buddy_models_deepseek_r1_server -- -j1
```

Building that plugin target does not import weights or generate model kernels.
Use matching runtime/plugin versions, artifact dimensions, weight layout,
kernel libraries, and vocabulary; repackage resident v2 plugins into existing
`.rax` files as described in [the migration guide](resident-v2.md). A configured
real model can then be served with:

```bash
build/bin/buddy-server --model "$DEEPSEEK_RAX" \
  --chat-template examples/BuddyDeepSeekR1/deepseek-r1.json
```

To check the packaged model over HTTP, run completion/chat, streaming, raw
codecs, and repeated requests with the actual artifact and sufficient memory.
Plugin builds and scripted-session tests do not exercise model inference.
