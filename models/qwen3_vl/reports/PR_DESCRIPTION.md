## Summary

- Align Qwen3-VL RAX path to **IEEE FP16** (import, shims, runner/runtime ABI).
- Optimize SpacemiT K3 single-image OCR via **RVV BLIS** tweaks, then **KV prefill/decode** + **packed decode GEMV** (`BUDDY_QWEN3_VL_KV_DECODE=ON` by default).
- Document measurement, `perf` hotspots, IME→SIGILL rationale, and before/after in `models/qwen3_vl/reports/K3_Qwen3VL_F16_RVV_StageTimers.md`.
- **IME / XSMTIME not enabled:** board probe shows `vfmadot` **SIGILL**; integer `smt.vmadot` works but needs an INT8 track. MatMul stays on RVV for this FP16 path.

**Headline (vs FP16 baseline, same OCR gate):** ~**6.13×** tok/s, e2e **−82%** (`opt-kv-decode-packed-f16`).

## Test plan

- [ ] Cross-build `qwen3_vl_rax` with `BUDDY_QWEN3_VL_KV_DECODE=ON`, OCR Pass on K3 (`Buddy MLIR` / `Qwen-3-VL 0.0` / `2026`).
- [ ] Confirm logs: `using KV prefill/decode` + `using packed decode weights`.
- [ ] Optional: `-DBUDDY_QWEN3_VL_KV_DECODE=OFF` still builds legacy `qwen3vl_decoder` for serving/Runtime.
- [ ] Do **not** enable `+xsmtime` / IME on X100 FP16 builds.

## Notes for reviewers

- **Default:** KV package; `Qwen3VLRunner` auto-detects prefill/decode.
- **Serving / `Qwen3VLRuntime`:** still `dlsym("qwen3vl_decoder")` only — use `KV_DECODE=OFF` until Runtime supports KV, or serving will miss the symbol on a KV-only `.so`.
- Report uses anonymized host/board labels; raw `k3_logs/` are not part of this PR.
