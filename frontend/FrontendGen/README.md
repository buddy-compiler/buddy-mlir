# How to build

FrontendGen is designed for generate mlir project quickly by writing fegen files.

The `FeGen` option needs to be enabled when building.

``` bash
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DFeGen=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
```