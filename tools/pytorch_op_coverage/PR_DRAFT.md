# Draft PR: PyTorch operator coverage MVP (#911)

## Summary
- Add reproducible PyTorch op coverage methodology (`docs/PytorchOpCoverage.md`).
- Define denominator **Buddy Target Op Set v0** (Transformer + MoE-critical seed).
- Add `tools/pytorch_op_coverage/` static analyzer + optional live probe mode.
- Commit first JSON/Markdown report artifacts (static mode).

## Non-claims
- Does **not** claim 90% live coverage. Static frontend+lowering evidence only.

## Test plan
- [x] `python tools/pytorch_op_coverage/run_coverage.py --out-dir tools/pytorch_op_coverage/out`
- [ ] `--mode live` on Linux/WSL with Buddy Python packages
- [ ] Reviewer confirms denominator + status taxonomy match issue #911 intent
