# buddy-server

`buddy-server` exposes model service HTTP APIs. A server process receives one
model package with `--model <file.rax>`; it does not accept standalone model or
plugin paths.

Each RAX package contains one model and may contain one server adapter:

```text
runner_library = "payload:<runner-plugin>"   # used by buddy-cli
model_library = "payload:<model-server-plugin>"
```

`model_library` is resolved by `ModelManifest`, including `file:` and embedded
`payload:` URIs. The server opens that shared library with `RTLD_NOW | RTLD_LOCAL`,
calls `buddy_model_plugin_descriptor_v1`, and validates the descriptor before
creating a model object.

The descriptor declares one `ModelServiceKind`: `Resident`, `Embedding`,
`MaskedLM`, or `Transcription`. After discovery, the server loads the existing
typed factory ABI for that service. The create/destroy symbols and model
interfaces are service-specific. Resident plugins require the
[v2 ABI](compatibility/resident-v2.md); the discovery descriptor, CLI runner,
and other service factories retain their existing ABI.

```bash
cmake --build build --target buddy-server
build/bin/buddy-server --model build/bin/deepseek_r1.rax
```

For a riscv64 Linux board, cross-compile the server with the toolchain and
runtime built by `make riscv`:

```bash
make riscv-server
build-cross-rv/bin/buddy-server --model /path/to/model.rax
```

Use `--host`, `--port`, `--model-type`, and `--chat-template` for configuration.
Routes are selected by the descriptor service kind:

* Resident: `/completion`, `/v1/chat/completions`, `/tokenize`
* Embedding: `/v1/embeddings`
* MaskedLM: `/v1/masked-lm`
* Transcription: `/v1/audio/transcriptions`

`runner_library`, `rhal.codeobj`, weights, tokenizer resources, and the
`buddy-cli` loading flow are unchanged. A new model in an existing service kind
only needs a model plugin exporting the descriptor and its existing typed
factory symbols; no new RAX library attribute is required.

## Build checks

With `BUDDY_ENABLE_TESTS=ON`, the resident adapter checks compile the production
sources even when every `BUDDY_BUILD_*_MODEL` option is OFF:

```bash
ninja -C build -j1 check-buddy-resident-adapters
ninja -C build -j1 check-buddy-server
ninja -C build -j1 check-buddy
```

`check-buddy-server` includes the adapter loading tests and the existing HTTP
tests, and `check-buddy` includes `check-buddy-server`. All server CTest checks
run with one worker.

| Link-check target | Production sources checked |
| --- | --- |
| `buddy-server-deepseek-resident-link-check` | DeepSeek resident model, plugin, and the real generated `ModelSession`. |
| `buddy-server-qwen3-vl-resident-link-check` | Qwen3-VL runtime, resident model, and plugin. |
| `buddy-server-llama31-tt-resident-portable-link-check` | TT execution, resident model, plugin, and NN backend with `BUDDY_LLAMA31_TT_HAS_TTNN=0`. |

DeepSeek uses the production f32 spec and a
[fixed local HF config](testdata/README.md) with `gen_config.py` and
`gen_session.py`; generated files stay in the build directory. These checks
do not import weights, generate model kernels, or require a model directory.
Linux links reject undefined symbols with `-z defs`. Each loading test uses the
production `RTLD_NOW` discovery and resident handles to validate the descriptor,
factory, and model type, then creates and destroys unloaded model instances.
The resident v2 load checks also query capabilities and verify that unsupported
raw decoding is rejected. The production adapters do not advertise compatible serving features;
their legacy entry points remain available.

These checks validate source compatibility and plugin linking. Inference
acceptance still needs real model artifacts. The portable TT check does not
validate the TTNN SDK branch: the separate production targets
`buddy_models_llama31_tt_server_runtime` and `buddy_models_llama31_tt_server`
require the actual TT SDK and compile with `BUDDY_LLAMA31_TT_HAS_TTNN=1`.

## Issue #912 compatibility contract

The comparison is pinned to Buddy
`5f4e6edaa07d284a34efdc4e47a2f888d042c8f2` and llama.cpp
`3e4dfa8f39f904e681737d9f512a6e6338d848aa`. The tables distinguish
the pinned Buddy baseline from the serving contract. Supported requests
depend on the loaded adapter and package. The examples in
[compatibility/](compatibility/README.md) specify request and response
semantics; synthetic fixtures do not execute model weights.

The upstream references are the fixed revision's
[endpoint documentation][llama-readme], [generation schema][llama-schema],
[schema alias handling][llama-schema-header], and
[HTTP tokenizer handlers][llama-context]. “Baseline” below means the pinned
Buddy source, rather than a result of running model weights. Test IDs identify
request and response contract scenarios.

| Interface | Fixed upstream behavior | Pinned Buddy baseline | Serving contract | Backend scope | Contract IDs and remaining differences |
| --- | --- | --- | --- | --- | --- |
| `GET /health` | Ready: 200; loading: 503. | Always 200; body distinguishes loading, ok, and error. | Ready: 200; unloaded/loading/error: 503; retain status and diagnostic message. | All service kinds. | H01–H04. Richer Buddy status body remains an extension. |
| `POST /completion` | Native completion supports several prompt shapes and JSON/SSE. | One nonempty string; `max_tokens` only; native stream also writes `[DONE]`. | One string; new-token budget; validated sampling/stops; JSON or terminal-event-plus-EOF SSE. | Resident only. | C01–C08, P01–P26, S01. Arrays/batches remain unsupported; native JSON is a documented subset. |
| `POST /v1/chat/completions` | Chat JSON and SSE. | JSON has no `created`; chunks lack role and use `finish_reason:""` before completion; terminal chunk always includes usage. | Preserve supported message roles/turns; apply template once; add `created`; role/content/finish chunks, optional separate usage chunk, one `[DONE]`. | Resident; supported roles, turns, images depend on loaded artifacts. | CH01–CH02, S02–S03. No tool calls or general Jinja engine. |
| `POST /tokenize` | Token IDs, optionally pieces; raw encoding. | `content`/`text`, `count_only`; DTO defaults `add_special=true`; DeepSeek/Qwen ignore that flag and add chat scaffolding. | Raw text; defaults `add_special=false`, `parse_special=true`, `with_pieces=false`; retain `text` and `count_only`. | Resident; nondefault modes require declared codec support. | T01–T03, P14. Tokenizer equivalence requires separate vocabulary/artifact validation. |
| `POST /detokenize` | Token array to `content`. | Route and resident method absent (404). | Validate IDs; join raw bytes; repair UTF-8 once; preserve spaces, special token text, and bytes after EOS. | Resident with raw decoding. | D01–D09. Buddy specifies replacement of malformed UTF-8 explicitly. |
| `GET /v1/models` | Single loaded-model entry outside router mode. | Route absent (404). | Single configured-model entry; ID agrees with health and generation. | All service kinds; no model switching. | M01, P13. No router, cache inventory, or upstream metadata parity. |
| `POST /v1/embeddings` | OpenAI-shaped embedding API includes more input shapes. | Embedding service only; one nonempty string; float output; input arrays/streaming rejected; `encoding_format` ignored. | Keep one-text float vectors; explicitly reject unsupported input shapes and encodings. | Embedding service only; resident models do not acquire embedding support. | E01–E03. Batches, token arrays, multimodal input, and base64 remain unsupported. |

Current behavior can be checked in [buddy-server.cpp](buddy-server.cpp),
[JsonCodec.cpp](JsonCodec.cpp),
[SimpleHttpServer.cpp](SimpleHttpServer.cpp), and
[ServingTypes.h](../../runtime/include/buddy/runtime/core/ServingTypes.h).
Existing executable coverage includes `buddy-server-json-codec`,
`buddy-server-model-service-http-integration`, startup-failure, plugin discovery, and
transcription tests. Contract IDs label request and response assertions in the
[contract specification](compatibility/README.md).

### Normalization and parameter contract

Process fields in this order: recognize names, normalize optional `null`, select
aliases, validate types and ranges, supply defaults, then check backend
capabilities. Remember which non-null fields were explicitly supplied. Required
prompt/content/messages/tokens remain subject to their endpoint input checks.
Unknown names, even with `null`, return 400. A recognized but unimplemented
optional feature is rejected when non-null; `null` does not request that feature.

| Field | Pinned Buddy parser | Contract default and accepted values | Contract IDs |
| --- | --- | --- | --- |
| Length aliases | Only `max_tokens`; other names ignored. | First non-null of `n_predict`, `max_completion_tokens`, `max_tokens`. Validate only that value; differing aliases are not a conflict. Default 512; integer `[-1, INT32_MAX]`. | P01–P03 |
| New-token budget | Backend-dependent: DeepSeek/Qwen use total-token limits; TT uses a new-token limit. | 0 evaluates the prompt without sampling; −1 runs until stop/context boundary; positive values count sampled new tokens. Prompt overflow is an error; no silent truncation/context movement. | C01–C08 |
| `temperature` | Number copied without range checks. | 0; finite number; nonpositive becomes greedy. | P04 |
| `top_k` | Integer narrowed to `int`. | 0; representable integer; nonpositive disables filtering. | P05 |
| `top_p` | Number copied without range checks. | 1; finite number in `(0,1]`. | P06 |
| `min_p` | Number copied without range checks. | 0; finite number in `[0,1]`. | P07 |
| `repeat_penalty` | Number copied without range checks. | 1; finite number greater than 0. | P08 |
| `repeat_last_n` | Integer copied without range checks. | 64; integer `[0, INT32_MAX]`; 0 disables; −1 rejected. Serving uses prompt plus generated history and penalizes each distinct ID once after choosing the window. | P09 |
| `seed` | Integer cast to `uint64_t`; Sampler treats 0 as random. | Omitted/`null`, −1, or 4294967295: random. `0..4294967294`: fixed seed. Other values: 400 without truncation. | P10–P12 |
| `stop` | Only string arrays collected; other types/elements ignored; adapters do not apply string matching. | String or string array; reject empty strings and invalid elements. Incremental matching handles overlaps, chunk boundaries, and UTF-8; matched text is withheld. | S01, P15 |
| `stop_token_ids` | Integer array entries collected without vocabulary validation. | Buddy extension: integer IDs within loaded vocabulary, merged with model defaults. | P16 |
| `stream` | Boolean, otherwise falls back to false. | false; boolean after optional-null normalization. | S01–S03 |
| `stream_options` | Ignored. | Object or `null`; `include_usage` defaults false, accepts bool/`null`. Unknown nested names rejected. Valid options accepted on nonstreaming requests without changing response shape. Top-level `include_usage` is unknown. | S03, P17–P19 |
| `model` | Chat handling varies: TT rejects a mismatch, DeepSeek/Qwen ignore it; embedding rejects a mismatch. | Optional string; does not switch the single loaded model; response uses actual configured ID. | P13, M01 |

In the pinned Buddy parser, wrong types often fall back to defaults and unknown
fields are generally ignored. Strict validation is a deliberate contract
difference, not a claim that upstream rejects identical inputs. In particular,
Buddy's seed bounds and `repeat_last_n=-1` rejection are stricter policies.
Upstream's seed sentinel is defined in [llama.h][llama-header] and interpreted
in [llama-sampler.cpp][llama-sampler].

Keep Buddy's greedy defaults (`temperature=0`, `top_k=0`, `top_p=1`, `min_p=0`)
and default budget 512. Upstream defaults differ. Keep the existing chat
`input`/`prompt`, tokenize `text`/`count_only`, and `image_path` extensions, with
proper type validation; when `messages` is provided, it takes precedence over
bare chat text. `add_special` adds model-defined special tokens only, never chat
wrapping. Capability checks must describe the loaded artifact/configuration,
not infer support from a model-family name.

### Response and streaming contract

Native final responses use `stop:true` for every completed request, including
length termination, with the actual `stop_reason`, usage, and timings. Upstream
uses `stop_type` values such as `eos`, `limit`, and `word`; Buddy currently uses
`stop_reason` values `stop`, `length`, `cancelled`, and `error`. Native field
names and timing names remain an explicit compatibility difference rather than
a promise of byte-for-byte upstream responses.

Prompt usage counts actual encoded prompt tokens. Completion usage counts
sampled tokens, including a token that triggers EOS or a stop. String/token
stops take precedence over the budget when both occur on the same step.
Generation is shared between streaming and nonstreaming serving paths; each
request resets its session, and cancellation/errors release resources so another
request can run. Serving does not change model weights,
kernels, or numerical precision.

Chat responses include `created` (Unix seconds). Every event for one request
uses the same ID, model, and timestamp. A normal chat stream sends an assistant
role event, content events with `finish_reason:null`, then a finish event. With
`stream_options.include_usage:true`, it adds a separate event with `choices:[]`
and usage. Exactly one `data: [DONE]` ends the stream. Without that option, no
usage event is sent. Native SSE instead ends after its final JSON event and EOF;
it has no `[DONE]` sentinel.

```text
# Contract chat SSE, abbreviated JSON; blank lines delimit events.
data: {"id":"chat-1","object":"chat.completion.chunk","created":1,"model":"fixture","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1,"model":"fixture","choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1,"model":"fixture","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1,"model":"fixture","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}

data: [DONE]

```

The error contract uses `{"error":{"message":"...","type":"...","code":400}}`,
with the offending parameter named in validation messages. It covers invalid
requests (400), unknown paths (404), wrong methods (405), unavailable models
(503), and unexpected failures (500). The pinned Buddy handler uses this
envelope; its HTTP router uses a different response and does not correctly
distinguish wrong methods. The contract requires validation before sending
headers. After SSE starts, report errors inside the stream and close the
connection. Check SSE startup/write results and cancel generation on
connection failure. A fake response or ready health probe does not exercise
those lifecycle paths.

### Adapter and regression boundaries

| Path | Adapter behavior | Validation requirements |
| --- | --- | --- |
| DeepSeek R1 | Shared Sampler and legacy generation loop; JSON chat template required for chat; old DeepSeek tokenizer adds scaffolding, including on `/tokenize`. | Raw codec and one template application; public serving control and supported sampling; real artifact completion/chat, JSON/SSE, cancellation and repeat-request checks. |
| Llama 3.1 TT | Separate tokenizer/template/execution; TTNN path samples host logits; portable fake path is also available. | Declare sampling only for a validated logits path; portable compile is separate from a build with real TTNN SDK and from device execution. |
| Qwen3-VL | Greedy decoder; one local image; resident chat concatenates message text and loses roles; tokenizer adds multimodal/chat scaffolding. | Keep supported greedy/single-image behavior; reject unsupported sampling or message shapes explicitly; preserve supported roles and turns. |
| CLI | Old bool `runGeneration` overload prints plain text or JSONL, with historical budget, repeat-penalty, and nonzero-seed behavior. | Golden tests execute that exact overload. The HTTP budget, seed, and history contracts do not apply to this overload. |
| Other service kinds | Dedicated embedding, masked-LM, and transcription factories/routes. | Preserve those interfaces, factories, and regression tests. |

The resident v2 loader requires
`buddy_create_resident_model_v2`, `buddy_destroy_resident_model_v2`, and
`buddy_resident_model_type_v2`, and checks all three before invoking a resident
entry point. It rejects v1 plugins without constructing an object. Rebuild
server/runtime/resident plugins together and repackage `.rax` files; there is no
v1 fallback. Discovery descriptor v1, CLI runner, embedding, masked-LM, and
transcription ABIs stay as they are. See the
[migration guide](compatibility/resident-v2.md) for the DTO boundary, checks, and
remaining backend acceptance requirements.
No claim of backend device acceptance, real inference, or external serving
integration follows from the contract examples or a no-weight adapter link check.

[llama-readme]: https://github.com/ggml-org/llama.cpp/blob/3e4dfa8f39f904e681737d9f512a6e6338d848aa/tools/server/README.md
[llama-schema]: https://github.com/ggml-org/llama.cpp/blob/3e4dfa8f39f904e681737d9f512a6e6338d848aa/tools/server/server-schema.cpp
[llama-schema-header]: https://github.com/ggml-org/llama.cpp/blob/3e4dfa8f39f904e681737d9f512a6e6338d848aa/tools/server/server-schema.h
[llama-context]: https://github.com/ggml-org/llama.cpp/blob/3e4dfa8f39f904e681737d9f512a6e6338d848aa/tools/server/server-context.cpp
[llama-header]: https://github.com/ggml-org/llama.cpp/blob/3e4dfa8f39f904e681737d9f512a6e6338d848aa/include/llama.h
[llama-sampler]: https://github.com/ggml-org/llama.cpp/blob/3e4dfa8f39f904e681737d9f512a6e6338d848aa/src/llama-sampler.cpp
