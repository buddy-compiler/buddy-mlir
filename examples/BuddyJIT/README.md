# Buddy JIT for Pytorch Module

This example shows how to convert a PyTorch `nn.Module` into MLIR code
and run it use the `ExecutionEngine`.

## Prerequisites

To run this example, you should build `bully-compiler` with MLIR Python Binding.

Firstly, create a python environment (optional) and install the necessary package.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Secondly, build the llvm with LLVM, MLIR and OpenMP.

```bash
cd buddy-mlir
mkdir llvm/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3)
ninja check-clang check-mlir omp
```

Then build the buddy mlir with the python binding.

```bash
cd buddy-mlir
mkdir build
cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
ninja
ninja check-buddy
```

And then set the `PYTHONPATH` to where the packages are built.

```bash
export PYTHONPATH=/path-to-buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}

# For example:
# Navigate to your buddy-mlir/build directory
cd buddy-mlir/build
export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

## Matrix Multiply Demo

The example `pytorch_matrix_multiplication.py` shows how to define a `nn.Module`,
convert it to MLIR code and run the MLIR using `ExecutionEngine`.

First, define a `nn.Module`.

```python
class MatrixMultiply(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)
```

Second, initialize the `DynamoCompiler` and import the `nn.Module`.

```python
# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp
)
a = torch.rand(2048, 2048, dtype=torch.float32)
b = torch.rand(2048, 2048, dtype=torch.float32)
dynamo_compiler.importer_by_export(MatrixMultiply(), a, b)
```

Lastly, get the function to run using `DynamoCompiler.dynamo_run()` method. Note that the return value is a list of
tensor.

```python
exec_func = dynamo_compiler.dynamo_run()
exec_func(a, b)[0]
```
This example will also validate the correctness of result and print the running time.

```bash
$ python examples/BuddyJIT/pytorch_matrix_multiplication.py
Is MLIR equal to Torch? True
MLIR time: 27291.54ms, Torch time: 325.96ms
```