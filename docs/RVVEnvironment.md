# Environment Setup Guide for MLIR and RVV Testing and Experiments

This guide provides instructions on setting up an environment to test the RISC-V Vector Extension using the buddy-mlir project.
The target platform for emulation is QEMU.

## Requirements

Before proceed any further make sure that you installed dependencies below

* [LLVM dependecies](https://llvm.org/docs/GettingStarted.html#requirements)
* [GNU Toolchain dependecies](https://github.com/riscv-collab/riscv-gnu-toolchain#prerequisites)
* [QEMU dependecies](https://wiki.qemu.org/Hosts/Linux)

## Build Steps

> **_NOTE:_** The build process includes several heavy stages. It may take significant time to clone and build all components.

0. Prepare `buddy-mlir` and Submodules

```
$ git clone https://github.com/buddy-compiler/buddy-mlir.git
$ cd buddy-mlir
$ git submodule update --init
```

1. Build Local LLVM/MLIR

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja check-clang check-mlir omp
$ export BUILD_LOCAL_LLVM_DIR=$PWD
```

2. Build Local `buddy-mlir`

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_RISCV_GNU_TOOLCHAIN=ON \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-buddy
$ export BUILD_RISCV_GNU_TOOLCHAIN_DIR=$PWD/thirdparty/riscv-gnu-toolchain/
$ export RISCV_GNU_TOOLCHAIN_SYSROOT_DIR=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}/sysroot/
```

3. Build Cross-Compiled Clang

```
$ cd buddy-mlir
$ mkdir llvm/build-cross-clang-rv
$ cd llvm/build-cross-clang-rv
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_TARGETS_TO_BUILD="RISCV" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_C_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang \
    -DCMAKE_CXX_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang++ \
    -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DLLVM_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/llvm-tblgen \
    -DCLANG_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/clang-tblgen \
    -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
    -DLLVM_TARGET_ARCH=RISCV64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ZSTD=Off
$ ninja clang lli
```

4. Build Cross-Compiled MLIR

```
$ cd buddy-mlir
$ mkdir llvm/build-cross-mlir-rv
$ cd llvm/build-cross-mlir-rv
$ cmake -G Ninja ../../llvm/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DCMAKE_CROSSCOMPILING=True \
    -DLLVM_TARGET_ARCH=RISCV64 \
    -DLLVM_TARGETS_TO_BUILD=RISCV \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_NATIVE_ARCH=RISCV \
    -DLLVM_HOST_TRIPLE=riscv64-unknown-linux-gnu \
    -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
    -DCMAKE_C_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang \
    -DCMAKE_CXX_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang++ \
    -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DMLIR_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-tblgen \
    -DLLVM_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/llvm-tblgen \
    -DMLIR_SRC_SHARDER_TABLEGEN_EXE=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-src-sharder \
    -DMLIR_LINALG_ODS_YAML_GEN=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-linalg-ods-yaml-gen \
    -DMLIR_PDLL_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-pdll \
    -DLLVM_ENABLE_ZSTD=Off
$ ninja
$ export BUILD_CROSS_MLIR_DIR=$PWD
```

5. Build Cross-Compiled `buddy-mlir`

```
$ cd buddy-mlir
$ mkdir build-cross-rv
$ cd build-cross-rv
$ cmake -G Ninja .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DMLIR_DIR=${BUILD_CROSS_MLIR_DIR}/lib/cmake/mlir \
    -DLLVM_DIR=${BUILD_CROSS_MLIR_DIR}/lib/cmake/llvm \
    -DCMAKE_CROSSCOMPILING=True \
    -DLLVM_TARGETS_TO_BUILD=RISCV \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_NATIVE_ARCH=RISCV \
    -DLLVM_HOST_TRIPLE=riscv64-unknown-linux-gnu \
    -DCMAKE_C_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang \
    -DCMAKE_CXX_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang++ \
    -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DLLVM_ENABLE_ZSTD=Off
$ ninja StaticMLIRCRunnerUtils StaticMLIRRunnerUtils
```

## Testing RVV Environment

```
$ cd buddy-mlir
$ cd examples/RVVDialect/
$ make rvv-mul-add-run

// Expected Output:
Unranked Memref base@ = 0x55555729aaa0 rank = 1 offset = 0 sizes = [20] strides = [1] data =
[0,  12,  26,  42,  60,  80,  102,  126,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
```

Congratulations! Your RVV environment is now fully set up. Enjoy exploring and testing!
