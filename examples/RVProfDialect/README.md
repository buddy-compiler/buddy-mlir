# RVProf Dialect Example
RVProf provides cycle-based profiling for RISC-V by inserting `rvprof.region_begin/end`, lowering them to runtime calls, and exporting a trace file.

## Examples

Run RVProf instrumentation.

```
make linalg-prof
```

Build a ELF with lowered RVProf calls and linked with `rd_cycle` runtime .

```
make e2e-build
```
