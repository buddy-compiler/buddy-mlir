if [ -z "$1" ]
then
  llvm_build_dir="llvm/build"
else
  llvm_build_dir="$1"
fi
if [ ! -d $llvm_build_dir ]
then
    mkdir $llvm_build_dir
fi

cd $llvm_build_dir
# assuming if there is something in llvm_build_dir, it is a valid build,
# so we won't build llvm
if [ -z "$(ls -A ./)" ]
then
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_BUILD_TYPE=RELEASE
    ninja
fi
cd ../..
mkdir build
cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../$llvm_build_dir/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../$llvm_build_dir/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
ninja check-buddy
