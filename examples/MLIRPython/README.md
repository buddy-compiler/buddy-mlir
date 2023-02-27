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

Custom Compiler from FX Graph to MLIR:
-------------------------------------------------------------------
opcode         name    target                   args       kwargs
-------------  ------  -----------------------  ---------  --------
placeholder    x       x                        ()         {}
placeholder    y       y                        ()         {}
call_function  add     <built-in function add>  (x, y)     {}
output         output  output                   ((add,),)  {}
-------------------------------------------------------------------
Printing the symbol table ...
x :  Value(<block argument> of type 'tensor<10xf32>' at index: 0)
y :  Value(<block argument> of type 'tensor<10xf32>' at index: 1)
add :  %0 = arith.addf %arg0, %arg1 : tensor<10xf32>
output :  %0 = arith.addf %arg0, %arg1 : tensor<10xf32>
-------------------------------------------------------------------
Printing the generated MLIR ...
module {
  func.func @generated_func(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}

-------------------------------------------------------------------
Bufferizing the module ...
#map = affine_map<(d0) -> (d0)>
module {
  func.func @generated_func(%arg0: memref<10xf32>, %arg1: memref<10xf32>) -> memref<10xf32> {
    %0 = bufferization.to_tensor %arg1 : memref<10xf32>
    %1 = bufferization.to_tensor %arg0 : memref<10xf32>
    %2 = bufferization.to_memref %0 : memref<10xf32>
    %3 = bufferization.to_memref %1 : memref<10xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
    %4 = bufferization.to_tensor %alloc : memref<10xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %2 : memref<10xf32>, memref<10xf32>) outs(%alloc : memref<10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    }
    %5 = bufferization.to_tensor %alloc : memref<10xf32>
    %6 = bufferization.to_memref %5 : memref<10xf32>
    return %6 : memref<10xf32>
  }
}

-------------------------------------------------------------------
Lowering the module to LLVM dialect ...
module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @generated_func(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<f32>, %arg6: !llvm.ptr<f32>, %arg7: i64, %arg8: i64, %arg9: i64) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(10 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(10 : index) : i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.null : !llvm.ptr<f32>
    %18 = llvm.getelementptr %17[10] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %19 = llvm.ptrtoint %18 : !llvm.ptr<f32> to i64
    %20 = llvm.mlir.constant(64 : index) : i64
    %21 = llvm.add %19, %20  : i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr<i8>
    %23 = llvm.bitcast %22 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<f32> to i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.sub %20, %25  : i64
    %27 = llvm.add %24, %26  : i64
    %28 = llvm.urem %27, %20  : i64
    %29 = llvm.sub %27, %28  : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr<f32>
    %31 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %32 = llvm.insertvalue %23, %31[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.insertvalue %15, %35[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %16, %36[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%12 : i64)
  ^bb1(%38: i64):  // 2 preds: ^bb0, ^bb2
    %39 = llvm.icmp "slt" %38, %13 : i64
    llvm.cond_br %39, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %40 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr %40[%38] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %42 = llvm.load %41 : !llvm.ptr<f32>
    %43 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.getelementptr %43[%38] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %45 = llvm.load %44 : !llvm.ptr<f32>
    %46 = llvm.fadd %42, %45  : f32
    %47 = llvm.getelementptr %30[%38] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %46, %47 : !llvm.ptr<f32>
    %48 = llvm.add %38, %14  : i64
    llvm.br ^bb1(%48 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return %37 : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  }
}
```
