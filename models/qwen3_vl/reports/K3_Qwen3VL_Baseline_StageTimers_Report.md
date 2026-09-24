# Qwen3-VL 2B RAX on SpacemiT K3 — F32 measurement archive

**Issue:** [#892](https://github.com/buddy-compiler/buddy-mlir/issues/892)
**Config label:** `baseline+stage-timers`
**Precision:** F32
**Date:** 2026-09-16 / 2026-09-17

> **Superseded for Code Review.**
> The complete #892 deliverable report (F32 stages + RSS, reference, `perf`,
> IME→SIGILL / RVV strategy, FP16 before/after, checklist) lives in:
>
> **[`K3_Qwen3VL_F16_RVV_StageTimers.md`](K3_Qwen3VL_F16_RVV_StageTimers.md)**
> (title: *Issue #892 Report*)

This file keeps the original F32 measurement write-up for archival detail.
Do not use it as the sole review entry point; §7–§8 below are historical
(pre-optimization) and are **outdated**.

---

## 1. Environment

| Item | Value |
|---|---|
| Compile host | x86_64 cross-compile host |
| Board | SpacemiT K3 / X100 (Bianbu 4.0.1, `riscv64`) |
| Threads | `OMP_NUM_THREADS=8`, `--cpus 0-7`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |
| Runtime | RISC-V `buddy-cli` + F32 `qwen3_vl.rax` (cross build) |
| Stage timers | `models/qwen3_vl/Qwen3VLRunner.cpp` → stderr `[stage] …` |
| Max RSS | Board has **no** `/usr/bin/time -v`; used `/proc/<pid>/status` `VmHWM` sampler + bash `time` wall |

Artifacts: `$BOARD_ROOT/baseline/qwen3_vl/`, deps `$BOARD_ROOT/deps/`.

---

## 2. Correctness gate

| Run | OCR golden (`Buddy MLIR` / `Qwen-3-VL 0.0` / `2026`) | Exit |
|---|---|---|
| Prior F32 baseline fg1–fg3 (2026-09-16) | Pass | 0 |
| Stage-timer `stage_correctness` (2026-09-17) | Pass | 0 |
| Stage-timer `stage_run{1,2,3}` | Pass | 0 |
| HF F32 reference (x86 host, 2026-09-17) | Pass | 0 |
| `perf` short runs (`max-tokens 3`) | Partial text `Buddy ML` (expected; truncated decode) | 0 |

Performance numbers below are only from OCR-pass full runs (`max-tokens 32`).

---

## 3. F32 baseline without stage splits

Three sequential runs on K3 (2026-09-16), pre-instrumentation package:

| Run | Wall clock | Decode s/token (approx) | Generated tokens | tokens/s |
|---|---|---|---|---|
| fg1 | ~60.1 min | ~167.7 | 20 | ~0.0060 |
| fg2 | ~59.3 min | ~165.5 | 20 | ~0.0060 |
| fg3 | ~58.7 min | ~163.7 | 20 | ~0.0061 |

Logs: `$BOARD_ROOT/logs/baseline_fg{1,2,3}.txt` (board-local; not in-tree).

---

## 4. Stage-timer baseline (`config: baseline+stage-timers`)

| Stage | What is timed |
|---|---|
| preprocess | C++ `preprocessImage` (or baked `pixel_values.bin` path) |
| vision_encoder | `qwen3vl_vision` shim |
| multimodal_merge | embed splice + deepstack scatter |
| decoder_total | full greedy decode loop |
| tokens_per_sec | `generated_tokens / (decoder_total_ms/1000)` |
| e2e_ms | preprocess start → OCR print |

### 4.1 Results (K3, 2026-09-17)

| Run | preprocess_ms | vision_ms | merge_ms | decoder_total_ms | tok/s | e2e_ms | wall (`time`) | peak VmHWM |
|---|---|---|---|---|---|---|---|---|
| correctness | 3.96 | 80456 | 0.79 | 3.521e6 | 0.00568 | 3.602e6 | 60m9s | 17.59 GiB |
| run1 | 5.65 | 82188 | 0.74 | 3.441e6 | 0.00581 | 3.523e6 | 58m43s | 17.59 GiB |
| run2 | 5.64 | 81324 | 0.77 | 3.421e6 | 0.00585 | 3.503e6 | 58m23s | 17.59 GiB |
| run3 | 5.61 | 80873 | 0.78 | 3.508e6 | 0.00570 | 3.590e6 | 59m50s | 17.59 GiB |
| **mean run1–3** | **5.63** | **81462** | **0.76** | **3.457e6** | **0.00579** | **3.538e6** | **~59.1 min** | **~17.59 GiB** |

**Breakdown (mean run1–3 vs e2e):** decoder ≈ **97.7%**, vision ≈ **2.3%**, preprocess+merge ≪ 0.01%.

---

## 5–6. Reference + profiling

Copied into the #892 total report §§3–4. See
[`K3_Qwen3VL_F16_RVV_StageTimers.md`](K3_Qwen3VL_F16_RVV_StageTimers.md).

---

## Historical checklist (2026-09-17 measurement phase only)

At measurement freeze, RVV/IME opts were still deferred. That status is **obsolete**;
optimization results and the updated checklist are in the total report §0 / §7.
