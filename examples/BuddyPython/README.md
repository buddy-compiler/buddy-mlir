# Buddy Compiler Python Importer
## Introduction
This package serves as the PyTorch importer of Buddy Compiler. It is built on top of TorchDynamo, a Python-level JIT compiler introduced in PyTorch 2.0. Using this importer, one can convert a PyTorch function/model to corresponding MLIR code.

## Quick Start

### Prerequisites
MLIR Python Bindings is required for this importer. Run below commands to build it.

```bash
## Build MLIR Python Bindings

Build MLIR Python Binding in Buddy-MLIR.

// [Option] Enter your Python virtual environment.
$ cd llvm
$ python3 -m pip install -r mlir/python/requirements.txt
$ cd build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=[path_to_python_executable]
$ ninja check-mlir
```

Add MLIR Python bindings to your Python path.
```bash
// In the LLVM build dirctory.
$ export PYTHONPATH=$(pwd)/tools/mlir/python_packages/mlir_core
```

Test the MLIR python bindings environment.

```python
$ python3
>>> from mlir.ir import Context, Module
>>> ...
```

### Demo
Run the following code to generate MLIR code for the `foo` function.
```python
import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

# Define the target function or model.
def foo(x, y):
    return x * y + x


# Define the input tensors
in1 = torch.randn(10)
in2 = torch.randn(10)

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

module, _ = dynamo_compiler.importer(foo, *(in1, in2))

print(module)
```
If everything works well, the output should be as below.
```mlir
module {
  func.func @forward(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %1 = tosa.add %0, %arg0 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    return %1 : tensor<10xf32>
  }
}
```

For more demos, please refer to [examples/BuddyPython](https://github.com/buddy-compiler/buddy-mlir/tree/main/examples/BuddyPython). We currently offer two demos below.

* `module_gen.py`: A more detailed version of the quick start demo.
* `bert.py`: Import a [bert-base-uncased](https://huggingface.co/bert-base-uncased) model, convert it to MLIR code.

## Methodology
[TorchDynamo](https://pytorch.org/docs/stable/dynamo/index.html) is a cutting-edge Python-level JIT compiler introduced in PyTorch 2.0, designed to make unmodified PyTorch programs faster. It achieves this by hooking into the frame evaluation API of CPython to rewrite the bytecode before it's executed. This process extract the sequences of PyTorch operations into a FX graph which is then just-in-time compiled with a compiler backend. While TorchInductor serves as the default backend, PyTorch 2.0 also offers an interface for custom compiler backends. This is the main entry point that help us implement this importer.

### Operator 

* **Operator Mappings**: What this importer do is to convert a piece of PyTorch code to the corresponding MLIR code. To achieve it, we write some conversion functions that map PyTorch's operators to MLIR code snippets. Currently, we've mapped about 20 operators. For what operators are supported, please refer to the [frontend/Python/ops](https://github.com/buddy-compiler/buddy-mlir/tree/main/frontend/Python/ops) directory.

* **Operator Registries**: We organize the operator mapping functions using operator registries. Each operator registry is a Python dict that maps the PyTorch operator's name to its corresponding mapping function. Currently, we've offer three operator registries, i.e. `tosa`, `math` and `linalg`. The registry name stands for the main MLIR dialect that used to implement a operator.


### Symbol Table
In PyTorch FX graph, there exist dependencies between operators. These dependencies represent the inputs and outpus of each operator. To handle the dependencies between operators and generate MLIR code for the whole FX graph, during the importing process, the importer will build a symbol table. This symbol table is a Python dict that maps the operator's name to the their corresponding MLIR operation. When a new PyTorch operator is going to be imported, the importer will search the symbol table for its inputs, i.e. the operator's argument(s), and the inputs' MLIR code snippet. After that, the importer will generate the MLIR code snippet for the operator and add it to the symbol table. This process will be repeated until the whole FX graph are imported.

### Import Strategy
In order to make the importing procedure more robust, we've implement a fallback importing strategy. This machenism is consisted of two parts, i.e. primary registry and fallback registry. When importer is going to import a PyTorch operator, it will first search the primary registry for the operator's mapping function. If the operator is not found in the primary registry, the importer will try to search the fallback registry. By default, the importer will use `tosa` registry as the primary registry, and all the other registries as the fallback registry.

## Limitations
Currently, we only support AOT execution of the generated MLIR code. To execute the generated MLIR code, one need to use the llvm tooltrain to compile it to an executable binary. We are working on the JIT execution of the generated MLIR code.
