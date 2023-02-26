#!/bin/bash
num_thread=""
if [ -n "$1" ]; then
  num_thread="$1"
  echo "Number of threads was set to $num_thread for make"
fi
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
else
  echo "riscv-gnu-toolchain was cloned already"
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
  make linux -j $num_thread
  cd ../..
else
  echo "riscv-gnu-toolchain was built already"
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
  make -j $num_thread
  cd ../..
else
  echo "qemu was cloned and built already"
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
else
  echo "native clang was built already"
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
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ZSTD=Off
  ninja clang lli
  cd ..
else
  echo "clang cross-compiler for riscv64 was built already"
fi

#-------------------------------------------------------------------------------
# Build cross MLIR
#-------------------------------------------------------------------------------

if [ ! -d "build-cross-mlir" ]
then
  mkdir build-cross-mlir
  cd build-cross-mlir
  cmake -G Ninja ../../llvm/llvm \
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
    -DCMAKE_C_COMPILER=$PWD/../build-local-clang/bin/clang \
    -DCMAKE_CXX_COMPILER=$PWD/../build-local-clang/bin/clang++ \
    -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=$PWD/../build-riscv-gnu-toolchain/sysroot --gcc-toolchain=$PWD/../build-riscv-gnu-toolchain" \
    -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=$PWD/../build-riscv-gnu-toolchain/sysroot --gcc-toolchain=$PWD/../build-riscv-gnu-toolchain" \
    -DMLIR_TABLEGEN=$PWD/../../llvm/build/bin/mlir-tblgen \
    -DLLVM_TABLEGEN=$PWD/../../llvm/build/bin/llvm-tblgen \
    -DMLIR_LINALG_ODS_YAML_GEN=$PWD/../../llvm/build/bin/mlir-linalg-ods-yaml-gen \
    -DMLIR_PDLL_TABLEGEN=$PWD/../../llvm/build/bin/mlir-pdll \
    -DLLVM_ENABLE_ZSTD=Off
  ninja
else
  echo "mlir for riscv64 was built already"
fi
