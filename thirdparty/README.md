# Third-Party Dependencies

## The `mimalloc` Allocator

### Build `mimalloc`

```
$ cd buddy-mlir/thirdparty/mimalloc
$ mkdir -p out/release
$ cd out/release
$ cmake -G Ninja ../.. -DCMAKE_INSTALL_PREFIX=$PWD
$ ninja install
```

### Use `mimalloc` in buddy-mlir

Assign the `mimalloc` installation address to the `BUDDY_MLIR_USE_MIMALLOC` CMake variable.

For example:

```
$ cd buddy-mlir/build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_USE_MIMALLOC=ON \
    -DMIMALLOC_BUILD_DIR=$PWD/../thirdparty/mimalloc/out/release
```

In the `CMakeLists.txt` file, link shared or static library by using:

```
target_link_libraries(myapp PUBLIC mimalloc)
target_link_libraries(myapp PUBLIC mimalloc-static)
```

## Tenstorrent tt-mlir

`thirdparty/tt-mlir` is an optional submodule for Buddy's TTIR -> TTNN ->
P150A flow. It is not required for normal Buddy builds.

Initialize it only when working on Tenstorrent support:

```
$ git submodule update --init --depth 1 thirdparty/tt-mlir
```

Build and runtime setup are documented in:

- `docs/TenstorrentEnvironment.md`
- `thirdparty/tt-mlir/docs/src/getting-started.md`
- `thirdparty/tt-mlir/docs/src/ttrt.md`
