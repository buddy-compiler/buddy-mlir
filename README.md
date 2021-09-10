# BUDDY MLIR

BUDDY MLIR  ❤️  Teachers

We prepared a teacher's day card. (If you can't see any words on the card, that's right! Please follow the steps below to solve the puzzle！)

![Teacher's Day Card](./examples/conv-opt/images/TeachersDay.png)

## How to see the words on the card?

Note: please use a Linux machine with OpenCV installed to play around.

### LLVM/MLIR Dependencies

Please make sure [the dependencies](https://mlir.llvm.org/getting_started/) are available on your machine.

### Clone and Initialize


```
$ git clone git@github.com:buddy-compiler/buddy-mlir.git
$ cd buddy-mlir
$ git submodule update --init
```

### Build LLVM/MLIR

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja
```

### Build buddy-mlir

 You should specify the `<ISA vector extension>` (e.g. avx512f).

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_CONV_OPT_EXAMPLES=ON \
    -DBUDDY_CONV_OPT_STRIP_MINING=256 \
    -DBUDDY_CONV_OPT_ATTR=<ISA vector extension>
$ ninja teachers-day
```

### Solve The Puzzle

```
$ cd bin
$ ./teachers-day ../../examples/conv-opt/images/TeachersDay.png Happy.png
```
Now, see the `Happy.png` 🎉
