# Buddy JIT for PyTorch Training

This example shows how to train a PyTorch model through Buddy MLIR JIT using
`torch.compile`. Both the forward and backward passes can be compiled to MLIR
and executed via the Buddy `ExecutionEngine`, while the optimizer step runs in
PyTorch as usual.

## Prerequisites

To run this example, you should build `buddy-mlir` with MLIR Python Binding.

Firstly, create a python environment (optional) and install the necessary packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
```

Secondly, build LLVM with MLIR and OpenMP.

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

Then build buddy-mlir with the Python binding.

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
export PYTHONPATH=/path-to-buddy-mlir/build/python_packages:${PYTHONPATH}

# For example:
# Navigate to your buddy-mlir/build directory
cd buddy-mlir/build
export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

## MLP Training Demo

The example `train_mlp.py` trains a two-layer MLP in three modes: pure PyTorch,
Buddy MLIR with forward-only compilation, and Buddy MLIR with full forward+backward
compilation.

First, define a two-layer MLP in `model.py`.

```python
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))
```

Second, initialize a `DynamoCompiler` and wrap it with `TorchCompileBackend`.
Set `compile_backward=True` to also compile the backward graph to MLIR.

```python
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    compile_backward=True,
)
model = torch.compile(model, backend=TorchCompileBackend(compiler), dynamic=False)
```

Lastly, run the standard PyTorch training loop. The compiled graphs execute
transparently via `ExecutionEngine`; the optimizer step runs in PyTorch as usual.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(model(x), labels)
    loss.backward()
    optimizer.step()
```

Run the demo:

```bash
$ python examples/BuddyJITTrain/train_mlp.py
```

All three modes use the same random seed and produce identical loss values:

```
=== Pure PyTorch (baseline) ===
  epoch   1/20  loss=1.3791
  ...
  epoch  20/20  loss=1.3848
  weights changed : True

=== Buddy MLIR JIT – forward only (compile_backward=False) ===
  epoch   1/20  loss=1.3791
  ...
  epoch  20/20  loss=1.3848
  weights changed : True

=== Buddy MLIR JIT – forward + backward (compile_backward=True) ===
  epoch   1/20  loss=1.3791
  ...
  epoch  20/20  loss=1.3848
  weights changed : True

All training runs completed.
```
