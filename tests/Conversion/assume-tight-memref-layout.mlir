// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -convert-elementwise-to-linalg \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -expand-strided-metadata \
// RUN:     -ownership-based-buffer-deallocation \
// RUN:     -buffer-deallocation-simplification \
// RUN:     -bufferization-lower-deallocations \
// RUN:     -convert-bufferization-to-memref \
// RUN:     -assume-tight-memref-layout \
// RUN: | FileCheck %s

module {
  func.func @kernel(%b : tensor<8960x1536xf32>) -> tensor<1x1536xf32> {
    %cst_2 = arith.constant 2.0 : f32
    %empty_0 = tensor.empty() : tensor<1x8960xf32>
    %a = linalg.fill ins(%cst_2 : f32) outs(%empty_0 : tensor<1x8960xf32>) -> tensor<1x8960xf32>

    %cst_4 = arith.constant 4.0 : f32
    %empty_2 = tensor.empty() : tensor<1x1536xf32>
    %c = linalg.fill ins(%cst_4 : f32) outs(%empty_2 : tensor<1x1536xf32>) -> tensor<1x1536xf32>

    %273 = linalg.matmul {cast = #linalg.type_fn<cast_signed>}
      ins(%a, %b : tensor<1x8960xf32>, tensor<8960x1536xf32>)
      outs(%c : tensor<1x1536xf32>) -> tensor<1x1536xf32>

    return %273 : tensor<1x1536xf32>
  }
}

// FileCheck that verify tightened layout is unit-strided on the trailing dim.
// CHECK-LABEL: func.func @kernel
// CHECK: memref.reinterpret_cast {{.*}} :
// CHECK-SAME: memref<8960x1536xf32, strided<[?, 1], offset: ?>>
