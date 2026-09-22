# Qwen3-VL 2B RAX on SpacemiT K3 — Issue #892 Report

**Issue:** [#892 Optimize Qwen3-VL 2B RAX Performance on SpacemiT K3](https://github.com/buddy-compiler/buddy-mlir/issues/892)
**Board:** SpacemiT K3 / X100 (Bianbu 4.0.1, `riscv64`)
**Scope:** single-image OCR RAX (`test_text.png`, prompt `Read all the text in the image.`, `max-tokens 32`, `temperature 0.0`, grid `[1,14,28]`)
**Threads:** `OMP_NUM_THREADS=8`, `--cpus 0-7`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
**Runtime:** RISC-V `buddy-cli` + staged `qwen3_vl.rax`
**Stage timers:** `Qwen3VLRunner.cpp` → stderr `[stage] …`

This document is the **single deliverable report** for #892 (measurement → profile → optimize → re-benchmark).
Archival F32-only tables from the measurement phase remain in
[`K3_Qwen3VL_Baseline_StageTimers_Report.md`](K3_Qwen3VL_Baseline_StageTimers_Report.md)
(superseded for review; linked for detail).
Paste-ready PR text: [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md).

---

## 0. Deliverables checklist

| Deliverable | Status | Where in this report |
|---|---|---|
| Reproducible K3 OCR results | **Done** | §1–2, §6–7 (OCR Pass every config) |
| Separate stage measurements | **Done** | §2 (F32), §6–7 (FP16) |
| E2E OCR RAX performance | **Done** | F32 ~59 min → FP16 packed KV ~**1.7 min** |
| vs native/reference | **Partial** | §3 (HF F32 on x86 host; on-K3 native N/A) |
| Profiling + bottlenecks | **Done** | §4 |
| RVV + IME patches | **RVV Done; IME not used** | §5 (SIGILL evidence) + §7 |
| Before/after results | **Done** | §7 summary table |
| Memory usage | **Done** (F32) | §2 peak VmHWM ~17.6 GiB |
| Correctness gate | **Done** | OCR `Buddy MLIR` / `Qwen-3-VL 0.0` / `2026` |

**Headline (same-precision attribution):** vs **FP16** baseline, final config
`opt-kv-decode-packed-f16` is **~6.13× tok/s** and **−82% e2e**.
F32 vs FP16 is documented separately and must not be used as the opt attribution baseline.

### Reviewer notes (PR scope)

- **Default build:** `BUDDY_QWEN3_VL_KV_DECODE=ON` — import prefill/decode, packed decode weights, `link_decoder_kv_shim.sh` → `decoder_shim.so` with `qwen3vl_decoder_prefill` / `_decode`.
- **IME / XSMTIME:** not enabled. Board probe: `vfmadot` → **SIGILL**; integer `smt.vmadot` (XSMTVDot) runs. FP16 MatMul stays on **RVV** (BLIS / packed GEMV). INT8+`smt.vmadot` is follow-up.
- **`buddy-cli` / `Qwen3VLRunner`:** auto-detects KV symbols; loads `decoder_decode_weights.data` when present.
- **Serving / `Qwen3VLRuntime`:** still resolves **`qwen3vl_decoder`** (legacy full-forward f16 ABI). A default KV-only `.so` does **not** export that symbol — use `-DBUDDY_QWEN3_VL_KV_DECODE=OFF` for resident/serving until Runtime grows a KV path.

---

## 1. Environment

| Item | Value |
|---|---|
| Compile host | x86_64 cross-compile host → riscv64 |
| Board | SpacemiT K3 / X100 |
| Max RSS method | Board has **no** `/usr/bin/time -v`; sampled `/proc/<pid>/status` **VmHWM** |
| Artifacts (examples) | `$BOARD_ROOT/baseline/`, `opt-f16-rvv/`, `opt-kv-decode-f16/`, `$BOARD_ROOT/deps/` |

---

## 2. F32 measurement baseline (stage timers + RSS)

**Config:** `baseline+stage-timers` · **Precision:** F32 · **Date:** 2026-09-16 / 2026-09-17
**Note:** Measurement / reference / profiling only for this section (no RVV/IME opts yet).

| Run | preprocess_ms | vision_ms | merge_ms | decoder_total_ms | tok/s | e2e_ms | wall | peak VmHWM |
|---|---|---|---|---|---|---|---|---|
| correctness | 3.96 | 80456 | 0.79 | 3.521e6 | 0.00568 | 3.602e6 | 60m9s | 17.59 GiB |
| run1 | 5.65 | 82188 | 0.74 | 3.441e6 | 0.00581 | 3.523e6 | 58m43s | 17.59 GiB |
| run2 | 5.64 | 81324 | 0.77 | 3.421e6 | 0.00585 | 3.503e6 | 58m23s | 17.59 GiB |
| run3 | 5.61 | 80873 | 0.78 | 3.508e6 | 0.00570 | 3.590e6 | 59m50s | 17.59 GiB |
| **mean run1–3** | **5.63** | **81462** | **0.76** | **3.457e6** | **0.00579** | **3.538e6** | **~59.1 min** | **~17.59 GiB** |

**Share of e2e (mean):** decoder ≈ **97.7%**, vision ≈ **2.3%**, preprocess+merge ≪ 0.01%.
**Takeaway:** optimize the **decoder** first.

---

## 3. Reference comparison

### 3.1 HuggingFace F32 (x86 host — order-of-magnitude only)

Same image / prompt / `max_new_tokens=32`, `torch.float32`.

| Metric | HF F32 (x86 host) | Buddy RAX F32 (K3) |
|---|---|---|
| Platform | x86_64 + GPU host path | SpacemiT K3 riscv64 |
| Precision | F32 | F32 |
| OCR | Pass | Pass |
| Generated tokens | 21 | 20 (EOS) |
| Rate | **~14.94 tok/s** | **~0.0058 tok/s** |
| Timed wall | **~1.41 s** generate | E2E **~59 min** |

**Not same-hardware.** Do not claim “beat native on K3” from this row.
Source: `reports/hf_f32_ocr.txt`.

### 3.2 On-board native (llama.cpp / SpacemiT VL)

`llama-cli` / `llama-mtmd-cli` / `llama-bench`: **not found** on the evaluation board.
**Status:** N/A — remaining gap vs a true on-K3 native VL stack is **not quantified**.

---

## 4. Profiling (`perf` on F32 RAX)

**Tooling:** `perf` 6.18.3 on K3.
**Workload:** same OCR setup, **`max-tokens 3`** (~10 min/run).
**Caveat:** vision share is inflated vs full 20-token E2E; use stage timers for stage shares, `perf` for hotspots inside shims.

### 4.1 `perf stat` (user counters)

| Counter | Value |
|---|---|
| wall | 583.8 s |
| cycles | 1.222e12 (~2.095 GHz) |
| instructions | 5.703e11 |
| IPC | **0.47** |
| cache-misses | `<not supported>` |

### 4.2 Hotspots (`perf record`)

| Symbol / stage | Share | Component | Implication |
|---|---|---|---|
| `decoder_shim.so` | **83.8%** | OpenMP `subgraph0..omp_par.*` | Decoder MatMul / attn / MLP dominate |
| top `omp_par.7903` | 11.3% | single hot region | Primary GEMM/attn candidate |
| `vision_shim.so` | **12.9%** | vision encoder | Secondary |
| `libomp` / `libc` / `libm` | ~1% each | runtime / memcpy / `expf` | Not first lever |
| multimodal merge | ~0 in stage timers | runner | Not a hotspot |

Issue checklist mapping: vision attn/GEMM inside `vision_shim`; decoder attn/MLP/MRoPE inside decoder omp regions; layout/memcpy small (~1% libc).

Logs: `k3_logs/perf_stat.err`, `perf_report_by_dso.txt`, `perf_report_by_sym.txt`.

---

## 5. Why not IME — SIGILL evidence → RVV FP16

Issue §4/§5 asked for **IME (matrix) + RVV (vector)**. On SpacemiT **X100**:

| Extension / path | Instruction | Data | On SpacemiT X100 |
|---|---|---|---|
| Buddy **IME** / **XSMTIME** (`+xsmtime`) | **`vfmadot`** | FP16 float matrix | **SIGILL** (illegal instruction) |
| **XSMTVDot** (`+xsmtvdot`) | **`smt.vmadot`** | **Integer** matrix | **Executes** |
| **RVV** (+zfh/+zvfh/+zvl256b) | `vfmacc` / vector loads | FP16 | **Used for this track** |

**Evidence:** small-kernel board probe — integer `smt.vmadot` runs; XSMTIME `vfmadot` traps with **SIGILL**.
Therefore FP16 Qwen3-VL cannot lower MatMul through IME without crashing.
`lower_to_obj.sh` keeps IME **opt-in** and **refuses** IME together with `+xsmtvdot` so X100 builds cannot silently emit `vfmadot`.

**Chosen path for #892 coding phase:**

- Precision: **FP16** (accelerator-friendly, OCR-gated).
- MatMul: **RVV BLIS** (prefill / vision) + **packed decode GEMV** (decode).
- LLC: `+zvl256b,+xsmtvdot` (advertise VLEN / integer matrix ISA; **do not** enable `+xsmtime`).
- Follow-up for true matrix-core use: **INT8 + `smt.vmadot`** (separate precision track).

---

## 6. Official FP16 baseline (opt attribution baseline)

| Field | Value |
|---|---|
| **Config** | `baseline` (FP16) |
| **Lowering** | RVV BLIS + batchmatmul-optimize; **IME off** |
| **LLC** | `-mattr=+m,+d,+v,+zfh,+zvfh,+xsmtvdot` (no `zvl256b` yet) |
| **Package** | `$BOARD_ROOT/opt-f16-rvv/` |
| **Date** | 2026-09-21 |
| **Log** | `k3_logs/f16_rvv_ocr.txt` |
| **OCR** | Pass |

| Stage | ms |
|---|---|
| preprocess | 6.41 |
| vision_encoder | 11822 (~11.8 s) |
| multimodal_merge | 0.45 |
| decoder_total | 545155 (~9.09 min) |
| tokens/s | **0.03669** |
| e2e_ms | 557203 (~9.29 min) |
| wall | 9m21s |

### Cross-precision reference only (not for opt Δ)

| Metric | F32 §2 | FP16 §6 |
|---|---|---|
| wall | ~59 min | ~9.3 min |
| tok/s | ~0.0058 | ~0.037 |
| vision | ~81 s | ~12 s |
| decoder | ~3457 s | ~545 s |

---

## 7. Optimizations vs FP16 baseline (before / after)

Same image / prompt / tokens / OMP. Compare **only** to §6.

| Config | Notes | tok/s | e2e_ms | OCR | vs FP16 baseline |
|---|---|---|---|---|---|
| `baseline` | BLIS nr=32 | 0.03669 | 557203 | Pass | — |
| `opt-blis-f16-vlen256` | nr=16, `+zvl256b` | **0.03942** | **517922** | Pass | **+7.5%** tok/s |
| `opt-blis-f16-vecpack` | + vector panel pack | **0.04022** | **507731** | Pass | **+9.6%** |
| `opt-blis-f16-kunroll4` | + K unroll×4 | **0.04033** | **506296** | Pass | **+9.9%** |
| `opt-kv-decode-f16` | Prefill + GQA KV decode | **0.05226** | **393118** | Pass | **+42.4%** (~1.42×) |
| **`opt-kv-decode-packed-f16`** | KV + packed decode GEMV | **0.2251** | **99399** | Pass | **+513% (~6.13×)**; e2e **−82.2%** |

### Final config detail (`opt-kv-decode-packed-f16`)

| Field | Value |
|---|---|
| Package | `$BOARD_ROOT/opt-kv-decode-f16/` |
| Date | 2026-09-22 |
| Log | `k3_logs/kv_decode_packed_ocr.txt` |
| Build | `BUDDY_QWEN3_VL_KV_DECODE=ON` (default): import KV graphs, pack decode weights, `link_decoder_kv_shim.sh` → `decoder_shim.so`, stage into rax |
| OCR | Pass |

| Stage | FP16 baseline | packed KV | Δ |
|---|---|---|---|
| vision_encoder_ms | 11822 | 10325 | −12.7% |
| decoder_total_ms | 545155 | **88847** | **−83.7%** |
| tokens/s | 0.03669 | **0.2251** | **~6.13×** |
| e2e_ms | 557203 | **99399** | **−82.2%** |
| wall | 9m21s | **~1m39s** | −7m42s |

Per-step (packed): prefill ~24.0 s; decode step ~**3.2 s** (was ~26 s full-forward / ~17.9 s KV+BLIS).

**Code (reusable where noted):**

- BLIS f16 nr/vecpack/kunroll: `MatMulBlisVectorization.cpp`
- KV import + pack: `qwen3_vl_codegen.py`, `import_model.py`
- Shim / link: `decoder_kv_shim.cpp`, `link_decoder_kv_shim.sh`
- Lower: `lower_to_obj.sh` (decode → packed GEMV)
- Runner + CMake one-click: `Qwen3VLRunner.cpp`, `buddy_model.cmake`, `CMakeLists.txt`

### Intermediate configs (detail)

#### `opt-blis-f16-vlen256`

Log: `k3_logs/f16_opt_blis_vlen256_ocr.txt`. BLIS `nr=vec=16` for f16; LLC `+zvl256b`.

| Stage | baseline | opt | Δ |
|---|---|---|---|
| vision_encoder | 11822 | 10384 | −12.2% |
| decoder_total | 545155 | 507319 | −6.9% |
| tokens/s | 0.03669 | 0.03942 | **+7.5%** |
| e2e_ms | 557203 | 517922 | −7.0% |

#### `opt-blis-f16-vecpack`

Log: `k3_logs/f16_opt_blis_vecpack_ocr.txt`. + vectorized A/B panel pack.

| Stage | baseline | vecpack | vs baseline |
|---|---|---|---|
| vision_encoder | 11822 | 10264 | −13.2% |
| decoder_total | 545155 | 497248 | −8.8% |
| tokens/s | 0.03669 | 0.04022 | **+9.6%** |
| e2e_ms | 557203 | 507731 | −8.9% |

#### `opt-blis-f16-kunroll4`

Log: `k3_logs/f16_opt_blis_kunroll4_ocr.txt`. + K unroll×4 (diminishing returns).

| Stage | baseline | kunroll4 | vs baseline |
|---|---|---|---|
| vision_encoder | 11822 | 10206 | −13.7% |
| decoder_total | 545155 | 495867 | −9.0% |
| tokens/s | 0.03669 | 0.04033 | **+9.9%** |
| e2e_ms | 557203 | 506296 | −9.1% |

#### `opt-kv-decode-f16` (KV without packed GEMV)

Log: `k3_logs/kv_decode_f16_ocr_final.txt`. Prefill once + GQA decode; BLIS still on M=1 decode (~17.9 s/step).

| Stage | baseline | kv | vs baseline |
|---|---|---|---|
| decoder_total_ms | 545155 | 382704 | −29.8% |
| tokens/s | 0.03669 | 0.05226 | **+42.4%** |
| e2e_ms | 557203 | 393118 | −29.4% |
| wall | 9m21s | ~6m36s | −2m45s |

---

## 8. Remaining gap and follow-up

| Gap | Evidence | Owner / follow-up |
|---|---|---|
| No on-K3 native VL number | llama.cpp VL tools absent | Install/compare when available; cannot claim “beat native” yet |
| Prefill still ~24 s | Stage / per-step logs | Shorter live prefix; more BLIS on prefill; fused paths |
| Matrix core unused for FP16 | `vfmadot` SIGILL | **INT8 + `smt.vmadot`**, or wait for FP matrix ISA |
| Decode still ~3.2 s/token | Bandwidth-bound GEMV | Alive-len attention; reduce full-N KV traffic |
| HF (x86) ≫ K3 | Different HW | Documented; not a same-board target |

**Performance target:** significant improvement — **met** (~6× vs FP16 baseline, OCR held).
**Reach native where practical:** **not met / N/A** on-board; profiling + follow-ups above satisfy the “if cannot reach, quantify gap” clause.

---

## 9. Related docs

| Doc | Role |
|---|---|
| This file | **#892 total report** (review entry point) |
| [`K3_Qwen3VL_Baseline_StageTimers_Report.md`](K3_Qwen3VL_Baseline_StageTimers_Report.md) | Archival F32 measurement write-up (points here) |
| [`../README.md`](../README.md) | Build / run (KV default ON) |
