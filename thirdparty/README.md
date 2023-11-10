# Third-Party Dependencies

## The `snmalloc` Allocator

### Build `snmalloc`

```
$ cd buddy-mlir/thirdparty/snmalloc
$ mkdir -p build
$ cd build
$ cmake -G Ninja .. 
$ ninja install
```
It may need new C++ feature to build, If there's problem, use the `clang` built within the buddy-mlir, just follow the instruction of LTO. and change the command to 
```
$ cmake -G Ninja ..\
        -DCMAKE_CXX_COMPILER=$PWD/../../../llvm/build/bin/clang++
```
### Use `snmalloc` in buddy-mlir

Assign the `snmalloc` installation address to the `BUDDY_MLIR_USE_SNMALLOC` CMake variable.

For example:

```
$ cd buddy-mlir/build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_USE_SNMALLOC=ON \
    -DSNMALLOC_BUILD_DIR=$PWD/../thirdparty/snmalloc/build
```

In the `CMakeLists.txt` file, link shared or static library by using:

```
target_link_libraries(myapp PUBLIC snmalloc-static)
```
