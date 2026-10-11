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
interfaces remain service-specific.

```bash
cmake --build build --target buddy-server
build/bin/buddy-server --model build/bin/deepseek_r1.rax
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
