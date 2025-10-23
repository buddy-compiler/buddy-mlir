# BUDDY MLIR

An MLIR-based compiler framework designed for a co-design ecosystem from DSL (domain-specific languages) to DSA (domain-specific architectures). ([Project page](https://buddy-compiler.github.io/))

## Getting Started

### LLVM/MLIR Dependencies

Please make sure [the dependencies](https://llvm.org/docs/GettingStarted.html#requirements) are available
on your machine.

### Clone and Initialize

```
$ git clone git@github.com:buddy-compiler/buddy-mlir.git
$ cd buddy-mlir
$ git submodule update --init llvm
```

### Prepare Python Environment

```
$ conda activate <your virtual environment name>
$ cd buddy-mlir
$ pip install -r requirements.txt
```

### Build and Test LLVM/MLIR/CLANG

```
$ cd buddy-mlir
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja check-clang check-mlir omp
```

If your target machine includes an NVIDIA GPU, you can add the following configuration:

```
-DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
-DMLIR_ENABLE_CUDA_RUNNER=ON \
```

### Build buddy-mlir

```
$ cd buddy-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-buddy
```

Set the `PYTHONPATH` environment variable to include both the LLVM/MLIR Python bindings and `buddy-mlir` Python packages:

```
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

If you want to test your model end-to-end conversion and inference, you can add the following configuration

```
$ cmake -G Ninja .. -DBUDDY_ENABLE_E2E_TESTS=ON
$ ninja check-e2e
```

## Examples

We provide examples to demonstrate how to use the passes and interfaces in `buddy-mlir`, including IR-level transformations, domain-specific applications, and testing demonstrations.

For more details, please see the [examples documentation](./examples/README.md).

## Contributions

We welcome contributions to our open-source project!

Before contributing, please read the [Contributor Guide](https://buddycompiler.com/Pages/ContributorGuide.html) and [Code Style](https://buddycompiler.com/Pages/Documentation/CodeStyle.html).

To maintain code quality, this project provides pre-commit checks:

```
$ pre-commit install
```

## How to Cite

If you find our project and research useful or refer to it in your own work, please cite the survey paper in which the Buddy Compiler design was first proposed:

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
