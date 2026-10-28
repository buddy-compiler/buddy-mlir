# Qwen3-VL resident serving

The Qwen3-VL resident adapter uses the shared `runServingGeneration` controller
through `ServingTokenSource`. It uses greedy, fixed-grid,
single-image model execution and full-sequence decoder recomputation. The
`Qwen3VLRunner` and `TextContainer` CLI paths use different tokenizer and
request-control semantics.

## Raw vocabulary and tokenizer limits

Packaging copies `examples/BuddyQwen3/vocab.txt` into the RAX `vocab_uri`.
Its 151669 ID/spelling pairs match the official Qwen3-VL-2B-Instruct tokenizer
at revision `89644892e4d85e24eaac8bacfd4f463576704203`. That correspondence does
not establish tokenizer equivalence. The line-based vocabulary omits the
canonical tokenizer's BPE merge rules, regex pre-tokenization, and NFC Unicode
normalization. See the fixed official [tokenizer JSON][qwen-tokenizer] and
[tokenizer configuration][qwen-config].

Raw encoding uses deterministic longest-score segmentation of raw byte pieces,
with lower token IDs breaking ties. It preserves spaces, newlines, NUL, and
the original Unicode bytes. It adds no chat/image wrapper, applies no model
context limit, and never truncates. It also differs from the legacy Qwen
tokenizer, which scores mapped UTF-8 spelling lengths and adds its own fixed
chat/image scaffold. Neither canonical nor legacy token IDs are certified.

For example, `antidisestablishmentarianism` encodes to
`15359,4243,33400,76,79680,72,67813` with the raw codec, versus
`517,84242,33400,478,8821,2142` with the fixed official tokenizer. Both token
sequences decode to the same word. Decomposed `e\u0301` remains `68,53839` in
the raw codec; canonical NFC changes it to precomposed `é`, ID `963`. These
examples characterize known differences, not real-model accuracy.

`/tokenize` defaults to `add_special=false`, `parse_special=true`, and
`with_pieces=false`. Qwen has no automatic BOS in this tokenizer profile, so
`add_special=true` is a supported no-op. Special parsing recognizes the 14
Qwen special spellings from `endoftext` through `video_pad` at their loaded
IDs. Disabling it encodes those spellings as ordinary text. Added tool,
thinking, and FIM tokens remain ordinary candidates. `count_only` omits both
IDs and pieces; requested pieces are repaired to UTF-8 independently for
display. Their concatenation may differ from decoding joined byte fragments.
Loading rejects empty, duplicate, or malformed UTF-8 vocabulary entries.

`/detokenize` validates every ID, including IDs after EOS, then joins all raw
bytes before one UTF-8 repair pass. It preserves leading/repeated spaces,
special-token text, NUL, and content after EOS. Split byte fragments that form
a character decode together; malformed bytes become U+FFFD and an incomplete
final prefix becomes one U+FFFD. Empty token input produces empty text. The
padded LM-head IDs beyond the loaded vocabulary cannot be selected by serving.

## Chat, images, and static package constraints

Default chat rendering preserves system/user/assistant roles, empty message
contents, and multiple turns, then adds the assistant generation prefix once.
At least one user message is required. The adapter rejects unsupported roles
instead of flattening them into text. A configured Buddy JSON template is
loaded and cached with the model, controls supported roles and rendering, and
contributes its string/ID stops. Templates using `system_in_first_user` accept
one leading system message immediately followed by a user message.

The supported image layout is one local image before text in one user message.
Place its `image_url` content part first. Text-before-image and interleaved
image/text layouts are rejected because the current DTO and renderer cannot
preserve them. Multiple occurrences are rejected even if they refer to the
same file. Top-level `image_path` attaches to the first user message. If no
image is supplied, the bundled image is also attached to that first user
message. Local paths and hostless `file:` URIs are accepted; remote/data URIs,
in-memory image bytes, video, and generic vision markers are unsupported.

The image block is `<|vision_start|>`, exactly 98 consecutive
`<|image_pad|>` tokens, then `<|vision_end|>`. Images are resized to the fixed
grid `[1,14,28]`. The package's RoPE tables and image embedding positions are
static: the complete encoded prompt must match the packaged prompt length,
and its image-pad positions must match `img_pos`. Matching only the total
length is insufficient. Chat can preserve a conversation that the current
package cannot execute; a changed prompt shape requires rebuilt positional
resources. Requests that violate these guards fail before vision/decoder
execution. No prompt truncation or context movement is performed.

`/completion` takes an already rendered prompt. It adds neither a chat template
nor an image block; provide the exact markers and packaged layout yourself.
For the default build-time instruction, a native request can be constructed
without writing out 98 repeated markers manually:

```bash
python3 - <<'PY' | curl -sS http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' --data-binary @-
import json
image = '<|vision_start|>' + '<|image_pad|>' * 98 + '<|vision_end|>'
prompt = ('<|im_start|>user\n' + image + 'Read all the text in the image.'
          + '<|im_end|>\n<|im_start|>assistant\n')
print(json.dumps({'prompt': prompt, 'max_tokens': 32}))
PY
```

This uses the bundled image. Both completion and chat still require matching
positional resources; a model built with another instruction/layout may reject
this example. See the [model setup and image-first chat example][qwen-model].

## Generation and request lifecycle

Generation is greedy only. Omit `temperature`, `top_k`, `top_p`, `min_p`,
`repeat_penalty`, `repeat_last_n`, and `seed` in HTTP requests. Explicit non-null
sampling fields are rejected even at neutral/default values such as
`temperature:0`. Omitted/null fields use defaults. Direct C++ serving callers
also receive errors for unsupported nondefault sampler settings.

Budget aliases use priority `n_predict > max_completion_tokens > max_tokens`.
Zero runs vision and decoder prefill without selecting a token; -1 runs until
stop/context; positive values count newly selected tokens. Selected EOS/stop
IDs count in completion usage and take precedence over simultaneous budget
exhaustion. Prompt usage includes the rendered role and image tokens. A full
context permits prefill without selection; oversized prompts fail.

Request and configured-template stops are combined with model
`<|im_end|>`/`<|endoftext|>` IDs. String stops match raw bytes before UTF-8
repair across token/code-point boundaries, withholding matched content.
Streaming and non-streaming share the same controller, stop handling, usage,
and incremental UTF-8 repair. Returning false from a callback stops further
callbacks; cancellation before the terminal chunk suppresses it. Requests
serialize kernel access
and release their buffers on success, cancellation, or exception so later
requests can reuse the runtime.

The request model alias does not switch the loaded model. The configured
display name or loaded artifact ID appears consistently in results and all
stream chunks, including terminal chunks.

## Build and validation boundaries

Build `buddy-server`, runtime, and resident plugins from the same sources and
repackage the plugin in each RAX. The resident v2 DTO field
`ChatMessage.imagesAfterText` preserves unsupported image ordering for explicit
rejection. It defaults to false for direct C++ callers. See the
[ABI guide](resident-v2.md).

The default model-options-OFF tests can be run with:

```bash
ninja -C build -j1 buddy-server-qwen3-vl-raw-codec-test \
  buddy-server-qwen3-vl-serving-test
ctest --test-dir build --output-on-failure -j1 \
  -R '^buddy-server-qwen3-vl-(raw-codec|serving)$'
```

The raw codec test uses the repository vocabulary and small malformed/synthetic
fixtures. The serving test loads fixture RAX resources through the production
runtime and calls actual `dlopen` vision/decoder functions with synthetic
arithmetic. It exercises tokenizer/template, budget/stop/usage, static-position,
image-layout, callback, and recovery behavior. This is runtime integration
coverage without model weights; it is not actual Qwen inference. These tests
are included in `check-buddy-server` and `check-buddy`.

`check-buddy-resident-adapters` separately runs unloaded production-plugin
link/load checks. A model-enabled build can compile the production plugin
without importing weights or building model kernels:

```bash
cmake --build "$QWEN_BUILD" --target buddy_models_qwen3_vl_server -- -j1
```

Use a build configured for Qwen3-VL with its matching local model metadata.
Production-plugin compilation and synthetic runtime/HTTP fixtures do not
validate model kernels, real weights, OCR quality, device execution, or external
serving integrations.

[qwen-tokenizer]: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/tokenizer.json
[qwen-config]: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/tokenizer_config.json
[qwen-model]: ../../../models/qwen3_vl/README.md
