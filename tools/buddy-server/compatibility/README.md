# Executable HTTP contract examples

[target-contract.json](target-contract.json) specifies requests and representative
response assertions for the serving contract.
`buddy-server-serving-http-contract` executes all examples through HTTP sockets
against the synthetic resident and embedding fixture plugins. It is included in
`check-buddy-server`. It exercises the production JSON/HTTP layers and shared
serving generation controller. It does not load real model weights.

The fixture is deliberately synthetic. Token IDs, byte pieces, scripted sampled
tokens, and embedding values belong only to these examples. They are not
DeepSeek, Qwen, or Llama token IDs. The fixture uses context length 16, prompt
`A` encoding to one token, generated tokens `A`, `B`, EOS, and an independently
configured embedding service profile. Its chat encoder returns one prompt token
so usage examples stay small; real template and tokenizer behavior needs
separate model-specific coverage. Raw encoding and raw decoding are separately
scripted here to exercise byte repair (including noncanonical token pieces).

The file has `format_version:1`, fixed comparison revisions, a shared `fixture`,
and an `examples` array. Every example has a stable ID, request, profile, and
expected response. Optional `setup` overrides fixture state or supplies an
observable fixture assertion. `prefill_calls`, `sample_calls`,
`expected_seed_policy`, `expected_rendered_messages`, and `template_applications`
are checked against backend observations recorded by the fixture. The
seed examples check how explicit seeds reach the backend; real seeded sampling
and repeat-request reproducibility require separate Sampler/runtime tests.

Before starting a server, the executor rejects unsupported format versions,
duplicate IDs, unknown profiles or `setup`/`expect` fields, and assertions that
do not match the JSON or SSE response branch.

The executor uses these assertion rules:

* `request.method`, `path`, and optional `json` define the HTTP request.
* `expect.status` and `content_type` check HTTP metadata.
* `json_subset` recursively matches object members; arrays have the listed
  length/order and recursively match their elements. Fields omitted from the
  subset, such as generated IDs and measured timings, are not fixed constants.
* `strings_contain` maps JSON pointers to required message substrings;
  `required_json_types` maps pointers to JSON types. Integers exclude booleans.
  `absent_json_pointers` requires those members to be absent.
* `sse_json_subsets` specifies JSON events in order. Chat examples require
  exactly those JSON events. S01 permits additional empty nonterminal events
  before its final event; it must never expose the stop string. `done_sentinels`
  counts literal `[DONE]` events, and `eof` requires the connection to close.
* `consistent_event_fields` requires the same values across all JSON events;
  `required_event_types` validates their types. `usage_events` counts events
  carrying usage; `concatenated_content` joins native content or chat
  `delta.content` in order.

Do not compare timestamps, timings, or arbitrary generated IDs as golden bytes.
Always parse actual wire JSON strictly before comparing decoded values. A NUL
in D09 must be JSON-escaped on the wire while surviving as a NUL after parsing.
For SSE, check event framing, not only the presence of `[DONE]` in a response.

| IDs | Contract covered |
| --- | --- |
| H01–H04 | Ready/loading/unloaded/error health status. |
| C01–C08 | 0/1/2/N/−1 budgets, EOS precedence and usage, prompt/context boundaries. |
| P01–P03, P24–P26 | Alias priority, null fallback/default, integer/range validation, unselected aliases. |
| P04–P09, P23 | Sampling types/ranges and greedy normalization. |
| P10–P12, P20–P22 | Explicit seed zero, random sentinels, seed bounds, null before capability checks. |
| P13–P19 | Single-model identity, unknown fields, stop validation, streaming options. |
| CH01–CH02 | Chat response metadata, messages priority, single template application. |
| T01–T03 | Raw tokenizer defaults, count-only/text extensions, special text. |
| D01–D09 | Byte joining, incomplete/invalid UTF-8, spaces/special tokens after EOS, bad IDs, JSON escaping. |
| M01 | Single configured-model listing. |
| E01–E03 | Single text/float embedding and explicit unsupported input/encoding errors. |
| S01–S03 | Native EOF, chat role/content/finish sequence, optional usage, one chat `[DONE]`. |
| R01–R04 | Structured 404/405/503/500 before streaming headers. |

The HTTP test adds repeated requests after cancellation/errors, client
connections reset after the first HTTP header byte or first SSE event, backend
callback cancellation and session reset observations, terminal events carrying
text, and midstream exceptions. Separate response-writer tests use a socketpair
with a closed peer to check SSE header and event write failures deterministically.
Separate runtime tests cover overlapping/Chinese stops, UTF-8 tails, repetition
history, context/budget boundaries, and fixed seeds. Production DeepSeek tests
exercise its raw codec and template with an injected session. Production TT
execution/resident tests use a scripted token source to cover greedy budgets,
raw bytes and special tokens, single template application, string/token stops,
cancellation, and reuse after errors. Their vocabulary and TT programs are
fixtures; they do not establish canonical Llama token equivalence or execute
the TTNN SDK branch. Qwen codec tests use its production vocabulary and record
known canonical segmentation/normalization differences. Qwen serving tests
load fixture RAX resources through the production runtime and execute actual
`dlopen` vision/decoder functions with synthetic arithmetic. They cover raw
codecs, greedy budgets/stops/usage, roles and empty turns, image ordering,
static-position guards, callback cancellation, and failure recovery; they do
not execute Qwen model weights or kernels. CLI golden tests execute
the legacy generation overload. See the [TT serving guide](llama31-tt.md) and
[Qwen serving guide](qwen3-vl.md) for supported modes and tokenizer/package
limits. Portable link checks and synthetic HTTP/runtime fixtures do not
exercise real model inference, SDK branches, or devices.
See the [server validation boundaries](../README.md#build-checks) for the
separate real-weight, TTNN SDK, device, and external-client acceptance areas.
