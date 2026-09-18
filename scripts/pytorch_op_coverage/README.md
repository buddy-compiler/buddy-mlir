# PyTorch operator coverage scripts

Internal evaluation for issue #911. See [methodology, commands and acceptance criteria](../../docs/PytorchOpCoverage.md).

`data/target_ops_v1.json` defines the denominator and schema snapshot;
`probes.py` defines input cases and representative blocks. `run_coverage.py`
produces JSON and Markdown in static, trace or live mode. `worker.py` isolates
execution; `classify.py` and `report.py` preserve separate evidence levels.

`test_coverage.py` covers accounting and failures without optional dependencies.
`test_probes.py` adds PyTorch fixture and adapter checks. The old v0 target is
retained only to document the migration.
