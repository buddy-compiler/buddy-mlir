## How to cross compile RISC-V rax target on x86_64 machine

0. Clone code and initialize
  Refer to steps 0. Prepare `buddy-mlir` and Submodules in the [docs/RVVEnvironment.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md).

1. Build and test LLVM/MLIR
  Refer to steps 1. Build Local LLVM/MLIR in the [docs/RVVEnvironment.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md).

2. Build buddy-mlir
  Refer to steps 2. Build Local `buddy-mlir` in the [docs/RVVEnvironment.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md) and add `export BUDDY_MLIR_BUILD_DIR=$PWD` `export PYTHONPATH=${BUILD_LOCAL_LLVM_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}` after execution.

3. Build Cross-Compiled MLIR
  Refer to steps 4. Build Cross-Compiled MLIR in the [docs/RVVEnvironment.md](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md).
  Note: Add `export RISCV_LLVM_BUILD_DIR=${BUILD_CROSS_MLIR_DIR}` `export RISCV_MLIR_C_RUNNER_UTILS=${BUILD_CROSS_MLIR_DIR}/lib/libmlir_c_runner_utils.so.22.0git` (the suffix is the LLVM version of the build) after execution.

4. Pull the OpenMP shared library of RISC-V
   Since the repository depends on OpenMP shared libraries, follow the steps below to set up the OpenMP dependency:
```bash
$ cd ${BUILD_LOCAL_LLVM_DIR}/../
$ wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XEsAhOcMioN9gdufuyO9OrHIdR0UtHh2' -O build-omp-shared-rv.tar.gz
$ mkdir build-omp-shared-rv && tar -xzf build-omp-shared-rv.tar.gz -C build-omp-shared-rv && rm build-omp-shared-rv.tar.gz
$ export RISCV_OMP_SHARED=${BUILD_LOCAL_LLVM_DIR}/../build-omp-shared-rv/libomp.so
```

5. generate rax file
```bash
cd buddy-mlir

python3 tools/buddy-codegen/build_model.py \
  --spec models/deepseek_r1/specs/f32.json \
  --build-dir build \
  --target deepseek_r1_rax \
  --is-rvv-crosscompile \
  --riscv-gnu-toolchain ${BUILD_RISCV_GNU_TOOLCHAIN_DIR} \
  --riscv-omp-shared ${RISCV_OMP_SHARED} \
  --riscv-mlir-c-runner-utils ${RISCV_MLIR_C_RUNNER_UTILS} \
  --riscv-llvm-build-dir ${RISCV_LLVM_BUILD_DIR} \
  --buddy-mlir-build-dir ${BUDDY_MLIR_BUILD_DIR} \
  -j 8
```

Everything the rax file carries is built for riscv64: the model library
(`deepseek_r1_model.so`), the OpenMP and `mlir_c_runner_utils` libraries above
together with the MLIR runtime libraries `mlir_c_runner_utils` needs
(`mlir_float16_utils`, `mlir_apfloat_wrappers`, taken from its directory), and
the runtime plugins named in its manifest
(`deepseek_r1_runner.so`, `deepseek_r1_serving.so`). The plugins are compiled
by `${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin/clang++` with the sysroot of the
GNU toolchain and linked with `lib/libLLVMSupport.a` and `lib/libLLVMDemangle.a`
of the cross-compiled LLVM (`--riscv-llvm-build-dir`), whose `include/` also
supplies the target `llvm/Config` headers. The same plugins are still built for
the host under `models/deepseek_r1/host/`, as the host tools link their static
library. Models with `RUNTIME_LINK_LIBS` (Whisper) and Qwen3-VL are not
supported by this path yet; configuring them with `--is-rvv-crosscompile` stops
with an error.

6. use
Transfer the generated rax file to the RISC-V machine and load it using buddy-cli to use it.
