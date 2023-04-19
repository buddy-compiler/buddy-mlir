# let the whole script failed if any command failed
set -e

if [ -z "$1" ]
then
  llvm_build_dir="llvm/build"
else
  llvm_build_dir="$1"
fi

if [ ! -d "llvm/build" ]
then
    # because all Makefiles in tests assume the build directory is called "llvm/build"
    # we get the absolute path of the build directory and create a symlink to it
    ln -s $(readlink -f $llvm_build_dir) llvm/build
fi

test_dirs=( \
  'BudDialect' \
  'DLModel' \
  'FrontendGen' \
  'MLIRAffine' \
  'MLIREmitC' \
  'MLIRGPU' \
  'MLIRLinalg' \
  'MLIRMath' \
  'MLIRMemRef' \
  'MLIRPDL' \
  'MLIRSCF' \
  'MLIRTensor' \
  'MLIRTOSA' \
  'MLIRVector' \
  'Pooling' \
)

cd "examples"
for subdir in "${test_dirs[@]}"; do
  echo "Testing $subdir"
  cd $subdir
  make test-all
  cd ..
done
