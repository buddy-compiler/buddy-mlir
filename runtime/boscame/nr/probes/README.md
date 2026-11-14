# NR capability probes

Diagnostic binaries only — they do **not** extend the production `nr_isa.py`
allowlist.

```bash
make -C runtime/boscame/nr/probes
examples/BOSCAMEFPGA/fpga_run.sh \
  runtime/boscame/nr/probes/build/rvv_probe.bin --fpga=<N>
```

| Image | Checks |
| --- | --- |
| `rvv_probe.bin` | Selected RVV ops / VL settings; illegal-instruction skip via temporary `mtvec` |
| `copy_probe.bin` | `nr_copy_bytes` length × alignment matrix with canaries |

`SUPPORTED` / `UNSUPPORTED` / `EXECUTED NUMERIC_FAIL` describe each case.
`verify RVV probe completion: PASS` means the probe finished, **not** that
every instruction is production-legal. Parent runtime: [`../README.md`](../README.md).
