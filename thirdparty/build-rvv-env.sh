#!/bin/bash

#-------------------------------------------------------------------------------
# Clone riscv-gnu-toolchain
#-------------------------------------------------------------------------------

if [ ! -d "riscv-gnu-toolchain" ]
then
  git clone git@github.com:riscv-collab/riscv-gnu-toolchain.git
  cd riscv-gnu-toolchain
  git checkout rvv-next
  git submodule update --init --recursive
  cd ..
fi

#-------------------------------------------------------------------------------
# Build riscv-gnu-toolchain
#-------------------------------------------------------------------------------

if [ ! -d "build-riscv-gnu-toolchain" ]
then
  cd riscv-gnu-toolchain
  mkdir build-linux
  cd build-linux
  ../configure --prefix=$PWD/../../build-riscv-gnu-toolchain
  make linux -j
  cd ../..
fi

#-------------------------------------------------------------------------------
# Clone and build QEMU for RVV
#-------------------------------------------------------------------------------

# TODO: test qemu in riscv-gnu-toolchain master branch

# cd ..
# mkdir build-qemu
# cd build-qemu
# ../configure --prefix=$PWD/../build-qemu
# make build-qemu -j

if [ ! -d "qemu" ]
then
  git clone git@github.com:sifive/qemu.git
  cd qemu
  git checkout 856da0e94f
  mkdir build
  cd build
  ../configure
  make -j
  cd ../..
fi

#-------------------------------------------------------------------------------
# Build local clang
#-------------------------------------------------------------------------------

if [ ! -d "build-local-clang" ]
then
  mkdir build-local-clang
  cd build-local-clang
  cmake -G Ninja ../../llvm/llvm \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DCMAKE_BUILD_TYPE=RELEASE
  ninja
  cd ..
fi

#-------------------------------------------------------------------------------
# Build cross clang and lli
#-------------------------------------------------------------------------------

if [ ! -d "build-cross-clang" ]
then
  mkdir build-cross-clang
  cd build-cross-clang
  cmake -G Ninja ../../llvm/llvm \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_TARGETS_TO_BUILD="RISCV" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_C_COMPILER=$PWD/../build-local-clang/bin/clang \
    -DCMAKE_CXX_COMPILER=$PWD/../build-local-clang/bin/clang++ \
    -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=$PWD/../build-riscv-gnu-toolchain/sysroot --gcc-toolchain=$PWD/../build-riscv-gnu-toolchain" \
    -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=$PWD/../build-riscv-gnu-toolchain/sysroot --gcc-toolchain=$PWD/../build-riscv-gnu-toolchain" \
    -DLLVM_TABLEGEN=$PWD/../build-local-clang/bin/llvm-tblgen \
    -DCLANG_TABLEGEN=$PWD/../build-local-clang/bin/clang-tblgen \
    -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
    -DLLVM_TARGET_ARCH=RISCV64 \
    -DCMAKE_BUILD_TYPE=Release
  ninja clang lli
  cd ..
fi
