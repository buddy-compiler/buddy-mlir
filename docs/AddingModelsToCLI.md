# Adding New Models to buddy-cli

This guide explains how to integrate new models with the **current buddy-cli + Buddy Runtime** stack, aligned with **`tools/buddy-codegen/`** (minimal JSON + generated boilerplate).

---

## 1. Three concepts to keep straight

| Concept | Role |
|--------|------|
| **`buddy-cli`** | Generic entry: reads **`.rax`**, parses **`model_name`**, **`makeRunner(modelName)`** builds the right **`InferenceRunner`**. |
| **`.rax`** | FlatBuffers-packed RHAL manifest: kernel **`.so`**, weight URIs, vocab, etc.; **`model_name` must match the runner registration** (below). |
| **`buddy_add_model`** (`tools/buddy-codegen/cmake/buddy_model.cmake`) | From **variant spec JSON**: `gen_config` → `gen_session` / `gen_manifest` → (optional) `import_model` → `compile_pipeline` → link **`.so`** → `rax-pack`. Today’s implementation is **shaped for autoregressive LLMs (prefill/decode + KV)**. |

**As of this repo:** `tools/buddy-cli/buddy-cli.cpp` **`makeRunner` only recognizes the prefix `deepseek_r1`** when CMake is configured with **`BUDDY_BUILD_DEEPSEEK_R1_MODEL=ON`** (linking **`buddy_models_deepseek_r1`**). Any other name fails with an error that points to how to extend the CLI.

---

## 2. Two kinds of work: new **variant** vs new **model family**

### 2.1 New precision / quantization in the **same** family (e.g. f16, bf16, w8a16)

**Goal:** stay on **DeepSeek R1 + buddy-codegen**; most boilerplate is already produced by **`gen_config.py`**, **`gen_session.py`**, **`gen_manifest.py`**, **`compile_pipeline.py`**.

1. Add a small **variant spec** under `models/deepseek_r1/specs/` describing:
   - `model_family`, `variant` (e.g. `f16`, `bf16`, `w8a16`)
   - optional `hf_model_path` for HuggingFace alignment
   - optional `weights_override` (element counts, etc.)
2. **`gen_config.py`** expands **`VARIANT_PRECISION`**, **`VARIANT_WEIGHT_TEMPLATES`**, and related tables into a full **`config.json`** (shapes, weight tags, MLIR/C++ types).
3. **Build:**
   `python3 tools/buddy-codegen/build_model.py --spec models/deepseek_r1/specs/<variant>.json`
   or configure CMake with **`BUDDY_DSR1_SPEC`**, **`BUDDY_BUILD_DEEPSEEK_R1_MODEL=ON`**, and run `ninja deepseek_r1_rax`.

**Relationship to `examples/BuddyDeepSeekR1`:** that tree often contains **legacy scripts / standalone CMake demos** (e.g. per-precision imports). **Prefer `models/deepseek_r1/specs/*.json` + `buddy_add_model` as the single source of truth**; when migrating, fold differences into **spec JSON + `gen_config` tables** instead of maintaining multiple hand-written pipelines.

---

### 2.2 A **new** model family (e.g. Llama, Whisper, MobileNet, Stable Diffusion)

There is **no** universal “change one JSON and everything works” path yet; split the problem into **build** and **runtime**.

#### (1) Runtime: what you must write by hand

1. Implement a subclass of **`buddy::runtime::InferenceRunner`** (see `DeepSeekR1Runner`):
   - Paths from manifest/CLI, **`dlopen`**, weight load, inputs, inference loop (or single forward), output.
2. Extend **`makeRunner`** in **`tools/buddy-cli/buddy-cli.cpp`**, e.g.
   `modelName.rfind("my_llm", 0) == 0` → `std::make_unique<MyModelRunner>()`
   and update the **Supported models** string in error messages.
3. In **`tools/buddy-cli/CMakeLists.txt`**, **`target_link_libraries(buddy-cli PRIVATE buddy_models_<name> ...)`** for your static library.
4. **`.rax` `model_name`** (RHAL `rhal.module` attribute) must match the same prefix convention, or the CLI cannot dispatch.

#### (2) Build side: recommended path toward “one JSON + codegen”

1. Create **`models/<your_model>/`** with at least:
   - **`CMakeLists.txt`** calling **`buddy_add_model(NAME ... SPEC ... RUNNER_SRC ...)`** (see `models/deepseek_r1/CMakeLists.txt`).
   - **Runner sources** (e.g. `YourModelRunner.cpp`) and **`include/buddy/runtime/models/YourModelRunner.h`** (hand-written; not from `gen_session`).
2. **`specs/<variant>.json`:** start from the DeepSeek spec shape, then change **`model_family`**, shapes, and weight layout; extend **`gen_config.py`** (or add **`gen_config_<model>.py`** and invoke it from CMake) to emit **`config.json`** for downstream tools.
3. **Generated code:**
   - **Autoregressive LLM** with a DeepSeek-like surface (KV, `forward_prefill` / `forward_decode`, multi-weight): **reuse / fork `gen_session.py`** or factor a shared generator.
   - **Non-LLM** (detection, diffusion, ASR): there is **no** full `gen_session`/`compile_pipeline` path yet; you need **new templates** or a **hand-written `ModelSession` equivalent**, then move stable pieces into generators.
4. **`import_model.py` / `compile_pipeline.py`** are tied to **DeepSeek graph layout and file naming**. New models need **extended import** or a **custom import script** wired into **`buddy_add_model`** Mode B/C (similar to `DEEPSEEKR1_MLIR_DIR` / import stamp)—this is the heaviest part and needs MLIR/compiler alignment.
5. **`gen_manifest.py`:** emits RHAL `.mlir`; if buffers/constants differ from DeepSeek, **extend or replace** manifest generation so **`rax-pack`** and **`ModelManifest::loadFromRax`** still resolve weights and `.so` URIs.

6. **CMake:** add **`add_subdirectory(<your_model>)`** in **`models/CMakeLists.txt`**.

---

## 3. Other deep-learning examples under `examples/`

Examples such as **BuddyLeNet, BuddyLlama, BuddyStableDiffusion, Whisper** are often **tutorials or standalone CMake flows** and may **not** use:

- **`.rax` + `buddy-cli`**, or
- **`buddy_add_model` + a single JSON spec**.

**Migration path:**

1. Pull tunables into **one JSON spec** (model id, shapes, quantization, weight list, compile flags).
2. Fold repeated CMake/Python into the **`buddy_model.cmake` pattern** or a **dedicated `compile_pipeline` variant**.
3. Implement **`InferenceRunner`** and register it in **`makeRunner`** to surface the model in the **unified CLI**.

Until (2)–(3) are done, those examples build from their own READMEs and are **not** wired to **`buddy-cli --model *.rax`**.

---

## 4. What usually belongs in the JSON spec (aligned with `gen_config`)

The idea is: **spec = family + variant + overrides**; merge the rest from HuggingFace **`config.json`** (when applicable) and **static tables** in code.

Typical DeepSeek spec fields (see `models/deepseek_r1/specs/*.json`):

- `model_family`, `variant`, `hf_model_path`
- `max_token_len`, `num_threads`
- Quantization: `quantization`, `weights_override` (per-tensor element counts, etc.)

For a **new family**, define a **custom spec schema** and teach **`gen_config`** how to expand it into **`config.json`** for manifest, session generation, and compilation.

---

## 5. Checklist before shipping a new model family

- [ ] **`InferenceRunner` subclass** implemented; **`run()`** paths match the manifest.
- [ ] **`buddy-cli.cpp`：`makeRunner`** updated; **`.rax` `model_name`** matches.
- [ ] **`buddy-cli` CMake** links **`buddy_models_<name>`**.
- [ ] **`models/CMakeLists.txt`** includes **`add_subdirectory`**.
- [ ] Build produces **`<name>_model.so`** and **`<name>.rax`** (or your family’s naming).
- [ ] Smoke test: **`buddy-cli --model build/models/<name>/<name>.rax ...`** (flags depend on model type; LLMs use `--prompt`, etc.).

---

## 6. File index

| Topic | Path |
|--------|------|
| CLI dispatch | `tools/buddy-cli/buddy-cli.cpp` (`makeRunner`) |
| Runner API | `runtime/include/buddy/runtime/core/InferenceRunner.h` |
| DeepSeek reference | `models/deepseek_r1/DeepSeekR1Runner.cpp`, `include/buddy/runtime/models/DeepSeekR1Runner.h` |
| CMake integration | `models/deepseek_r1/CMakeLists.txt`, `tools/buddy-codegen/cmake/buddy_model.cmake` |
| JSON → full config | `tools/buddy-codegen/gen_config.py`, `models/deepseek_r1/specs/*.json` |
| Session / manifest gen | `tools/buddy-codegen/gen_session.py`, `gen_manifest.py` |
| MLIR compile | `tools/buddy-codegen/compile_pipeline.py` |
| PyTorch → MLIR | `tools/buddy-codegen/import_model.py` |
| One-shot build | `tools/buddy-codegen/build_model.py` |
| Runtime overview | `docs/Runtime.md` |

---

## 7. Summary

- **DeepSeek variants only:** add **`specs/<variant>.json`** and extend **`gen_config.py`** tables as needed; **no CLI dispatch change**.
- **New model family:** implement and register **`InferenceRunner`**, and connect the build to **`buddy_add_model`** or **parallel codegen**; **migrate legacy `examples/`** if you want “single JSON + codegen + buddy-cli”.
- **End state:** **minimal JSON** for deltas, **generate** repeated code; **DeepSeek R1** is the first full reference; other architectures follow the same checklist incrementally.
