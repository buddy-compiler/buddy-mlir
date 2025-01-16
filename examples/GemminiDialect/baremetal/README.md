## Gemmini-Baremetal-Example

This example demonstrates how to use the gemmini dialect in a baremetal environment. Functional evaluation can be performed using spike, while RTL hardware-based evaluation uses verilator.

Compiling baremetal workloads requires ``riscv64-unknown-elf-gcc``. The riscv-toolchain in the gemmini chipyard environment already includes ``riscv64-unknown-elf-gcc``, so there's no need to reinstall it. Before running the following test cases, simply switch to the corresponding chipyard **conda environment**.

### Function Evaluation
Simply execute the generated executable file for testing, for example:
```
make mvin-mvout-run-baremetal
spike --extension=gemmini mvin-mvout-baremetal
```

### Hardware Evaluation
Step 1. Execute `make <workload>-run-baremetal` in this folder to generate the baremetal test case.

Step 2. Copy the generated test case `<workload>-baremetal` to the gemmini workload storage path in the chipyard folder (`chipyard/generators/software/gemmini-rocc-tests/build/`)

Step 3. `cd chipyard/generators/gemmini/` and execute `./scripts/run-verilator.sh <workload>` to run the test. (this step may be different depending on the chipyard/gemmini version)

It is strongly recommended to use chipyard's MEMORY_LOAD setting, which will significantly speed up the simulation startup time.
