# Setting up enviroment for testing MLIR RVV dialect

This guide will help to set up enviroment for testing RISC-V Vector Extention using buddy-mlir project and 
corresponding RVV Dialect. As a target platform QEMU emulator is used.

## Requirements

Before proceed any further make sure that you installed dependecies  below

* [LLVM dependecies](https://llvm.org/docs/GettingStarted.html#requirements)
* [GNU Toolchain dependecies](https://llvm.org/docs/GettingStarted.html#requirements)
* [QEMU dependecies](https://wiki.qemu.org/Hosts/Linux)

## Build steps

1. Clone buddy-mlir project
``` bash
git clone git@github.com:buddy-compiler/buddy-mlir.git
cd buddy-mlir
git submodule update --init
```
> **_NOTE:_** `buddly-mlir` contains `llvm-project` as a submodule. `llvm-project` is large, so cloning will take a while 

2. Run a script building enviroment
```
cd buddy-mlir/thirdparty
./build-rvv-env.sh
```
> **_NOTE:_** The scripts consist of multiple heavy stages, so be patient - it will take a while to clone and build 
everything.
Detailed description of the steps can be found in [the page](https://gist.github.com/zhanghb97/ad44407e169de298911b8a4235e68497)

> **_NOTE:_** By default, the script allows `make` to use all available threads for compilation. It may lead 
to consuming a lot of memory and crashing the compiler. If you face with the issue, try to limit the number of threads 
by passing an argument to the script. For example, `./build-rvv-env.sh 4`
