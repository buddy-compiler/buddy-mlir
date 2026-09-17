# Triton frontend for the NR FPGA examples

This resource builds the actual Python `@triton.jit` frontend and the
`triton-riscv` TTIR-to-linalg conversion against this repository's Buddy/LLVM.
Generated linalg is passed to the local Buddy NR/RVV lowering, then linked with
the same bare-metal launch and validation code as the handwritten linalg cases.
The frontend does not require a GPU or access to the FPGA server.

## Build

Build this checkout's LLVM and Buddy first, including its NR FPGA changes.
Activate the Python 3.12 environment, then run from the repository root:

```bash
conda activate boscame
examples/FPGA-BOSCAME/common/triton/setup-triton.sh --jobs=24
source examples/FPGA-BOSCAME/common/triton/triton-env.sh
```

`thirdparty/triton-riscv` is a submodule pinned to
`git@github.com:CBalaa/triton-riscv-for-fpga.git`, the plugin fork containing
the maintained LLVM API compatibility patch. Initialize it with
`git submodule update --init thirdparty/triton-riscv`; the setup script can
also clone it if absent. The script fetches the upstream Triton frontend into
the plugin's `triton` subdirectory, at the exact commit in
`toolchain-lock.json`. This nested Triton checkout is managed by the setup
script, not registered as a submodule. SSH access to GitHub is required for
SSH clone URLs. Existing checkouts must already match the lock: their branches
and edits are preserved.

The script applies the plugin's four locked patches idempotently and verifies
the resulting source contents by replaying the series from the pristine
upstream revision. Build dependencies are installed in the selected Python
environment. `--skip-deps` skips that installation if they are already present.

Python is selected from the active environment; an explicit path also works:

```bash
TRITON_PYTHON=/path/to/environment/bin/python \
LLVM_SYSPATH=/path/to/llvm/build \
BUDDY_MLIR_BINARY_DIR=/path/to/buddy/build/bin \
  examples/FPGA-BOSCAME/common/triton/setup-triton.sh
```

`triton-env.sh` discovers the repository-relative `llvm/build-2d26` and
`build-migrate/bin`, falling back to `llvm/build` and `build/bin`. Explicit
environment values take precedence. It also exports `TRITON_SHARED_OPT_PATH`
for the frontend conversion. Source it after each new shell activation.

## Verification and provenance

```bash
examples/FPGA-BOSCAME/common/triton/setup-triton.sh --check
```

This verifies commits, patch hashes and applied patches, then compiles two real
JIT kernels through `ASTSource`, `CPUBackend.make_ttir`, and the triton-riscv
conversion: an M=1 signed i8 dot and a floating-point RMS reduction.
`--check` does not clone, patch, install packages, rebuild, or access a board.

`toolchain-lock.json` records source commits, Python version, patches with
SHA-256 hashes, and source constraints. Patch verification replays from the
locked Triton revision (or an explicit `base_commit`, if recorded) and compares
the resulting file contents with the checkout. The nested checkout therefore
contains the expected patch changes after setup; these changes are reproduced
from the plugin's versioned patch files. A lock/provenance update that preserves
the source contents does not require rebuilding an existing frontend; run
`--check` to verify its provenance and smoke tests. A fresh checkout still needs
the normal build above. `requirements-build.txt` pins the
Python build dependencies used for the successful build. PyTorch is optional
for this offline frontend; the separate upstream execution driver requires it.
The tested boscame environment already contained `torch==2.10.0+cpu`.

The extra patch adds MLIR symbol traits and the new LLVM `InlineAsmOp`
`convergent` argument. `TRITON_SHARED_BUILD_TRITON_SAN=OFF` disables the unrelated
upstream sanitizer component, whose LLVM APIs target a different revision.
Neither change substitutes handwritten linalg for the Triton frontend.

Only the model architecture is taken from the Qwen reference repository.
Its v0.1 operator implementation is not used. Current NR instruction encodings
and boot/link conventions follow ModelZoo and are maintained under this
example's `common` and `tools`; the runnable examples do not import reference
repository code.
