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
// RUN:     -matmul-vectorization-decode \
// RUN:     -batchmatmul-optimize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -affine-parallelize \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
// RUN:     -cse \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
  func.func private @rtclock() -> f64

  func.func @kernel(%a : tensor<12x1x1024xf32>, %b : tensor<12x1024x128xf32>) -> tensor<12x1x128xf32> {
    %t_start = call @rtclock() : () -> f64

    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %res = tosa.matmul %a, %b, %a_zp, %b_zp : (tensor<12x1x1024xf32>, tensor<12x1024x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x1x128xf32>

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
    return %res : tensor<12x1x128xf32>
  }

  func.func @main(){
    %cst_2 = arith.constant 2.0 : f32
    %empty_0 = tensor.empty() : tensor<12x1x1024xf32>
    %a = linalg.fill ins(%cst_2 : f32) outs(%empty_0 : tensor<12x1x1024xf32>) -> tensor<12x1x1024xf32>

    %cst_4 = arith.constant 4.0 : f32
    %empty_2 = tensor.empty() : tensor<12x1024x128xf32>
    %b = linalg.fill ins(%cst_4 : f32) outs(%empty_2 : tensor<12x1024x128xf32>) -> tensor<12x1024x128xf32>

    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c5 step %c1 {
      %res = func.call @kernel(%a, %b) : (tensor<12x1x1024xf32>, tensor<12x1024x128xf32>) -> tensor<12x1x128xf32>
      %tensor_unranked = tensor.cast %res : tensor<12x1x128xf32> to tensor<*xf32>

      // Print results.
      // call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
    }
    return
  }
}
