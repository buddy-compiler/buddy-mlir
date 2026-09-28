# qwen3-0.6b NR operator suite

Linalg operators for the NR FPGA. Each operator directory has `kernel.mlir`
and `launch.c`, builds through [`common.mk`](common.mk), and links
[`../common/nr`](../common/nr/README.md) (not `../common/runtime`).

**This table is the only operator inventory.** When you add a new operator,
add one row here (and the directory); do not update the parent
[`../README.md`](../README.md) Examples list.

| Operator | Status | Notes |
| --- | --- | --- |
| [`add_1x1024/`](add_1x1024/) | Working | `1x1024` f32 elementwise add |

## Build / run pattern

From the repository root (needs `buddy-opt` / `buddy-translate` in
`build/bin`, or set `BUDDY_BIN`):

```bash
make -C examples/BOSCAMEFPGA/qwen3-0.6b/<op> all check
```

`all` writes `<op>/build/<op>.bin`. `check` is a **host oracle** only; it
does not replace a board run.

```bash
export FPGA_SSH_HOST=<ssh-alias-or-host>
export FPGA_REMOTE_DIR=<uvhs-workdir>

examples/BOSCAMEFPGA/fpga_run.sh \
  examples/BOSCAMEFPGA/qwen3-0.6b/<op>/build/<op>.bin \
  --fpga=<N> --capture-seconds=900
```

Success on UART: `verify <op>: PASS`. Shared launcher notes:
[`../README.md`](../README.md), [`../tools/README.md`](../tools/README.md).

## Pipeline (`add_1x1024`)

Via `common.mk`:

1. `buddy-opt --lower-linalg-to-boscame=target=nr-fpga` stamps the NR contract
2. `convert-linalg-to-loops` lowers the elementwise add
3. `tools/ame_to_word.py` → `restrict_fpga_assembly.py` → link with `nr.ld`
4. `tools/check_nr_elf.py` audits the linked ELF

Do not link `common/runtime` into an operator image. Shared helpers:
[`support.h`](support.h) / [`support.c`](support.c). Details:
[`../common/nr/README.md`](../common/nr/README.md).

## Layout

```text
qwen3-0.6b/
├── common.mk              # Shared operator build
├── support.h / support.c  # MemRef / workspace / print_check
├── tools/
│   ├── host_main.c        # Host `main` for make check
│   ├── build_suite.py     # Config stamp and lowering audit for common.mk
│   └── vectorize_nr.py    # Copies the kernel through for kind=add
└── <op>/                  # One directory per operator (see table above)
    ├── makefile           # include ../common.mk
    ├── kernel.mlir
    ├── launch.c
    └── metadata.json
```

## See also

- Tree overview: [`../README.md`](../README.md)
- NR NH/RA runtime: [`../common/nr/README.md`](../common/nr/README.md)
- Encode / restrict / ELF audit: [`../tools/README.md`](../tools/README.md)
