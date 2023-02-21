# MLIR Python Bindings Examples

## Build MLIR Python Bindings

Build MLIR Python Binding in Buddy-MLIR.

```
// [Option] Enter your Python virtual environment.
$ cd llvm
$ python3 -m pip install -r mlir/python/requirements.txt
$ cd build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
$ ninja check-mlir
```

Add MLIR Python bindings to your Python path.

```
// In the LLVM build dirctory.
$ export PYTHONPATH=$(pwd)/tools/mlir/python_packages/mlir_core
```

Test the MLIR python bindings environment.

```
$ python3
>>> from mlir.ir import Context, Module
>>> ...
```

## MLIR as TorchDynamo Custom Backend

Enter your Python environment with MLIR Python bindings.
[Install PyTorch](https://pytorch.org/) preview(nightly) version.

```
// Navigate to the MLIR Python examples directory.
$ cd buddy-mlir/examples/MLIRPython

// [Option] Enter your Python virtual environment.

// Run the examples.
(mlir)$ python3 arith_add.py

-------------------------------------------------------------------
opcode         name    target                   args       kwargs
-------------  ------  -----------------------  ---------  --------
placeholder    x       x                        ()         {}
placeholder    y       y                        ()         {}
call_function  add     <built-in function add>  (x, y)     {}
output         output  output                   ((add,),)  {}
-------------------------------------------------------------------
Generating placeholder operation...
Generating placeholder operation...
Parsing a call_function operation...
Generating add operation...
Generating return operation...
-------------------------------------------------------------------
Printing the symbol table ...
x :  %0 = "placeholder"() : () -> f32
y :  %1 = "placeholder"() : () -> f32
add :  %2 = arith.addf %0, %1 : f32
-------------------------------------------------------------------
Printing the generated MLIR ...
module {
  %0 = "placeholder"() : () -> f32
  %1 = "placeholder"() : () -> f32
  %2 = arith.addf %0, %1 : f32
}
```
