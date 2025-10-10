# Alternative Build Methods

This document describes additional build configurations and alternative methods for building `buddy-mlir` beyond the standard approach outlined in the main README.

## Table of Contents

- [Build with Image Processing Libraries](#build-with-image-processing-libraries)
- [One-Step Build Strategy](#one-step-build-strategy)
- [Build with Nix](#build-with-nix)
- [Tools](#tools)

## Build with Image Processing Libraries

To configure the build environment for using image processing libraries, add the following options to your cmake configuration:

```bash
$ cd buddy-mlir/build
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

**Configuration Options:**

- `BUDDY_MLIR_ENABLE_DIP_LIB=ON`: Enables the Digital Image Processing (DIP) library
- `BUDDY_ENABLE_PNG=ON`: Enables PNG format support

## One-Step Build Strategy

If you want to use `buddy-mlir` tools and integrate them more easily into your projects, you can use the one-step build strategy. This method builds LLVM, MLIR, and `buddy-mlir` together as an LLVM external project.

```bash
$ cd buddy-mlir
$ cmake -G Ninja -Bbuild \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_EXTERNAL_PROJECTS="buddy-mlir" \
    -DLLVM_EXTERNAL_BUDDY_MLIR_SOURCE_DIR="$PWD" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    llvm/llvm
$ cd build
$ ninja check-mlir check-clang
$ ninja
$ ninja check-buddy
```

## Build with Nix

This repository provides Nix flake support for reproducible builds. Follow the [Nix installation instructions](https://nixos.org/manual/nix/stable/installation/installation.html) and enable [flake features](https://nixos.wiki/wiki/Flakes#Other_Distros.2C_without_Home-Manager) to set up Nix on your system.

### Development Environment

If you want to contribute to this project, enter the development shell:

```bash
$ nix develop .
```

This command sets up a bash shell with `clang`, `ccls`, `cmake`, `ninja`, and other necessary dependencies to build `buddy-mlir` from source.

### Binary Tools

If you only want to use the `buddy-mlir` binary tools:

```bash
$ nix build .#buddy-mlir
$ ./result/bin/buddy-opt --version
```

This approach provides a fully isolated and reproducible build environment without affecting your system configuration.

## Tools

### buddy-opt

`buddy-opt` is the optimization driver for `buddy-mlir`, similar to `mlir-opt` in LLVM. It provides access to all dialects and optimization passes defined in the `buddy-mlir` project.

**Usage:**

```bash
$ buddy-opt [options] <input-file>
```

**Common Options:**
- `--help`: Display available passes and options
- `--pass-pipeline`: Specify a custom pass pipeline
- `--mlir-print-ir-after-all`: Print IR after each pass

**Example:**

```bash
$ buddy-opt --lower-affine --convert-linalg-to-loops input.mlir
```

### buddy-lsp-server

`buddy-lsp-server` is a drop-in replacement for `mlir-lsp-server`, providing Language Server Protocol (LSP) support for all dialects defined in `buddy-mlir`.

**Features:**
- Code completion for custom dialects (`rvv`, `gemmini`, `dip`, etc.)
- Real-time error diagnostics
- Hover information and symbol navigation
- Syntax highlighting support

**Setup for VSCode:**

Modify the MLIR LSP server path in your VSCode settings:

```json
{
    "mlir.server_path": "/path/to/buddy-mlir/build/bin/buddy-lsp-server"
}
```
