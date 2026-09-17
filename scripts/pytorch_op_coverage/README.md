# PyTorch operator coverage scripts

Internal scripts that measure Buddy-MLIR PyTorch operator coverage for
[issue #911](https://github.com/buddy-compiler/buddy-mlir/issues/911).

Methodology and denominator rules: [`docs/PytorchOpCoverage.md`](../../docs/PytorchOpCoverage.md).

## Run

```bash
# From repository root
python scripts/pytorch_op_coverage/run_coverage.py \
  --out-dir scripts/pytorch_op_coverage/out

# Optional live probes (needs Buddy Python packages on PYTHONPATH)
python scripts/pytorch_op_coverage/run_coverage.py --mode live \
  --out-dir scripts/pytorch_op_coverage/out
```

## Layout

| Path | Role |
| --- | --- |
| `run_coverage.py` | CLI entry |
| `parse_frontend.py` | Parse `_ops_map` / `ops_registry` from sources |
| `classify.py` | Coverage status classification |
| `report.py` | JSON / Markdown emitters |
| `probes.py` | Optional live DynamoCompiler probes |
| `data/target_ops_v0.json` | Coverage denominator (Target Op Set v0) |
| `out/` | Generated reports |
