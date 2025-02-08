# Get the LLVM build directory
# Check if the first argument is provided,
# if not, set llvm_build_dir to the default value "llvm/build".
if [ -z "$1" ]
then
  llvm_build_dir="llvm/build"
else
  llvm_build_dir="$1"
fi
# Check if the specified directory for LLVM build exists, if not, create it.
if [ ! -d $llvm_build_dir ]
then
    mkdir $llvm_build_dir
fi

# Navigate to the LLVM build directory.
cd $llvm_build_dir
ls
pwd
ls ..

# Build and check the LLVM project.
# If cached build is available, it will save time by only checking the project.
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
ninja check-clang check-mlir omp

# Navigate back to the root project directory.
cd ../..
# Create the build directory for the project.
pwd
ls
mkdir build
cd build

# Build and check the buddy-mlir project.
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../$llvm_build_dir/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../$llvm_build_dir/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
ninja check-buddy
