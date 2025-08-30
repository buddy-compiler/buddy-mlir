# BUDDY MLIR

An MLIR-based compiler framework designed for a co-design ecosystem from DSL (domain-specific languages) to DSA (domain-specific architectures).
([Project page](https://buddy-compiler.github.io/)).

## Getting Started

The default build system uses LLVM/MLIR as an external library. 
We also provide a [one-step build strategy](#one-step) for users who only want to use our tools.
Please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available on your machine.

### LLVM/MLIR Dependencies

Before building, please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available
on your machine.

### Clone and Initialize

```
$ git clone git@github.com:buddy-compiler/buddy-mlir.git
$ cd buddy-mlir
$ git submodule update --init
```

### Build and Test LLVM/MLIR/CLANG
```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja check-mlir check-clang
```

If your target machine includes a Nvidia GPU, you can use the following configuration:

```
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
```

To enable MLIR Python bindings, please use the following configuration:

```
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
```

If your target machine has lld installed, you can use the following configuration:

```
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
```

### Build buddy-mlir

If you have previously built the llvm-project, you can replace the $PWD with the path to the directory where you have successfully built the llvm-project.

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja
$ ninja check-buddy
```

To utilize the Buddy Compiler Python package, please ensure that the MLIR Python bindings are enabled and use the following configuration:

```
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-buddy
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

To configure the build environment for using image processing libraries, follow these steps:

```
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_DIP_LIB=ON \
    -DBUDDY_ENABLE_PNG=ON
$ ninja
$ ninja check-buddy
```

To build buddy-mlir with custom LLVM sources:

```
$ cmake -G Ninja .. \
    -DMLIR_DIR=PATH/TO/LLVM/lib/cmake/mlir \
    -DLLVM_DIR=PATH/TO/LLVM/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLLVM_MAIN_SRC_DIR=PATH/TO/LLVM_SOURCE
```

<h3 id="one-step">One-step building strategy</h3>

If you only want to use our tools and integrate them more easily into your projects, you can choose to use the one-step build strategy.

```
$ cmake -G Ninja -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_EXTERNAL_PROJECTS="buddy-mlir" \
    -DLLVM_EXTERNAL_BUDDY_MLIR_SOURCE_DIR="$PWD" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    llvm/llvm
$ cd build
$ ninja check-mlir check-clang
$ ninja
$ ninja check-buddy
```

### Use nix

This repository have nix flake support. You can follow the [nix installation instruction](https://nixos.org/manual/nix/stable/installation/installation.html) and enable the [flake features](https://nixos.wiki/wiki/Flakes#Other_Distros.2C_without_Home-Manager) to have nix setup.

- If you want to contribute to this project:

```bash
nix develop .
```

This will setup a bash shell with `clang`, `ccls`, `cmake`, `ninja`, and other necessary dependencies to build buddy-mlir from source.

- If you want to use the buddy-mlir bintools

```bash
nix build .#buddy-mlir
./result/bin/buddy-opt --version
```

## Dialects

### Bud Dialect

Bud dialect is designed for testing and demonstrating.

### DIP Dialect

DIP dialect is designed for digital image processing abstraction.

## Tools

### buddy-opt

The buddy-opt is the driver for dialects and optimization in buddy-mlir project. 

### buddy-lsp-server

This program should be a drop-in replacement for `mlir-lsp-server`, supporting new dialects defined in buddy-mlir. To use it, please directly modify mlir LSP server path in VSCode settings (or similar settings for other editors) to:

```json
{
    "mlir.server_path": "YOUR_BUDDY_MLIR_BUILD/bin/buddy-lsp-server",
}
```

After modification, your editor should have correct completion and error prompts for new dialects such as `rvv` and `gemmini`.

### pre-commit checks

The .pre-commit-config.yaml file checks code format and style on each commit, using tools such as clang-format, black, and flake8. You can also run these checks without committing by using "pre-commit run --all-files". This ensures consistent coding standards and prevents common errors before pushing changes.

To get started, you should install pre-commit (e.g., pip install pre-commit) and verify that clang-format, black, and flake8 are available. On Linux, you can use your package manager for clang-format, and pip for Python tools. If you need to revert any unwanted formatting changes, you can use "git stash" or "git restore ." (for all files) or "git restore <file>" (for a specific file), or revert the commit through your Git history.

## Examples

The purpose of the examples is to give users a better understanding of how to use the passes and the interfaces in buddy-mlir. Currently, we provide three types of examples.

- IR level conversion and transformation examples.
- Domain-specific application level examples.
- Testing and demonstrating examples.

For more details, please see the [documentation of the examples](./examples/README.md).

## How to Cite

If you find our project and research useful or refer to it in your own work, please cite our paper as follows:

```
@article{zhang2023compiler,
  title={Compiler Technologies in Deep Learning Co-Design: A Survey},
  author={Zhang, Hongbin and Xing, Mingjie and Wu, Yanjun and Zhao, Chen},
  journal={Intelligent Computing},
  year={2023},
  publisher={AAAS}
}
```

For direct access to the paper, please visit [Compiler Technologies in Deep Learning Co-Design: A Survey](https://spj.science.org/doi/10.34133/icomputing.0040).
