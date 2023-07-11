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
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=[path_to_python_executable]
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

### Methodology
[TorchDynamo](https://pytorch.org/docs/stable/dynamo/index.html) is a cutting-edge Python-level JIT compiler introduced in PyTorch 2.0, designed to make unmodified PyTorch programs faster. It achieves this by hooking into the frame evaluation API of CPython to rewrite the bytecode before it's executed. This process extract the sequences of PyTorch operations into a FX graph which is then just-in-time compiled with a compiler backend. While TorchInductor serves as the default backend, PyTorch 2.0 also offers an interface for custom compiler backends. This is the main entry point that help us integrate TorchDynamo and Buddy Compiler.

In this demo, we first employ [AOT Autograd](https://pytorch.org/functorch/nightly/notebooks/aot_autograd_optimizations.html) to lower FX graph in Torch level IR to ATen/Prims IR.  This simplifies our task of implementing operator mappings with MLIR. Then, a `FXGraphImporter` is called in order to build a symbol table and generate an MLIR module. The symbol table is a Python dict that record the mapping relationship between operators and MLIR operations. During the import process, MLIR operations are generated dynamically according to each ATen/Prims operator's semantics. After the importing, an MLIR module with high-level MLIR dialects is generated, and it will be passed to the `Lowering` function in order to lower it to lower level dialects. 

The ultimate objective is to boost the speed of model training and inference through the integration of Buddy Compiler and TorchDynamo. By now, we have finished a basic importer and the lowering process. we've also mapped some ATen/Prims operators. The task of generating the compiled callable is still in process. Feel free to run our current demo to try the importer, lowering and operator mappings.

### How to run

Enter your Python environment with MLIR Python bindings.
[Install PyTorch](https://pytorch.org/) preview(nightly) version.

```
// Navigate to the MLIR Python examples directory.
$ cd buddy-mlir/examples/MLIRPython

// [Option] Enter your Python virtual environment.

// Run the examples.
(mlir)$ python3 arith_add.py

Custom Compiler from FX Graph to MLIR:
-------------------------------------------------------------------
opcode         name    target           args              kwargs
-------------  ------  ---------------  ----------------  --------
placeholder    arg0_1  arg0_1           ()                {}
placeholder    arg1_1  arg1_1           ()                {}
call_function  add     aten.add.Tensor  (arg0_1, arg1_1)  {}
output         output  output           ((add,),)         {}
Printing the generated MLIR
module {
  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}

-------------------------------------------------------------------
Bufferizing the module ...
#map = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: memref<10xf32>, %arg1: memref<10xf32>) -> memref<10xf32> {
    %0 = bufferization.to_tensor %arg1 : memref<10xf32>
    %1 = bufferization.to_tensor %arg0 : memref<10xf32>
    %2 = bufferization.to_memref %0 : memref<10xf32>
    %3 = bufferization.to_memref %1 : memref<10xf32>
    %alloc = memref.alloc() : memref<10xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %2 : memref<10xf32>, memref<10xf32>) outs(%alloc_0 : memref<10xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %6 = arith.addf %in, %in_1 : f32
      linalg.yield %6 : f32
    }
    %4 = bufferization.to_tensor %alloc_0 : memref<10xf32>
    %5 = bufferization.to_memref %4 : memref<10xf32>
    return %5 : memref<10xf32>
  }
}

-------------------------------------------------------------------
Lowering the module to LLVM dialect ...
module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(64 : index) : i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.mlir.constant(10 : index) : i64
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.mlir.null : !llvm.ptr
    %17 = llvm.getelementptr %16[10] : (!llvm.ptr) -> !llvm.ptr, f32
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.add %18, %12  : i64
    %20 = llvm.call @malloc(%19) : (i64) -> !llvm.ptr
    %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
    %22 = llvm.sub %12, %13  : i64
    %23 = llvm.add %21, %22  : i64
    %24 = llvm.urem %23, %12  : i64
    %25 = llvm.sub %23, %24  : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %27 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.insertvalue %20, %27[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.insertvalue %15, %29[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.insertvalue %14, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %13, %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%15 : i64)
  ^bb1(%33: i64):  // 2 preds: ^bb0, ^bb2
    %34 = llvm.icmp "slt" %33, %14 : i64
    llvm.cond_br %34, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %35 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.getelementptr %35[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %37 = llvm.load %36 : !llvm.ptr -> f32
    %38 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.getelementptr %38[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.load %39 : !llvm.ptr -> f32
    %41 = llvm.fadd %37, %40  : f32
    %42 = llvm.getelementptr %26[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %41, %42 : f32, !llvm.ptr
    %43 = llvm.add %33, %13  : i64
    llvm.br ^bb1(%43 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return %32 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
}

```
Currently supported operators include: arith\_add, addmm.
