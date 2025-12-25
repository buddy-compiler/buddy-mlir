// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -convert-elementwise-to-linalg \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s



#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
func.func private @rtclock() -> f64
func.func @kernel(%q : tensor<1x12x1x128xf32>, %k_cache : tensor<1x2x1024x128xf32>, %v_cache : tensor<1x2x1024x128xf32>, %mask : tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32> {

   %89 = tosa.reshape %k_cache {new_shape = array<i64: 1, 2, 1, 1024, 128>} : (tensor<1x2x1024x128xf32>) -> tensor<1x2x1x1024x128xf32>
    %90 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x1024x128xf32>}> : () -> tensor<1x2x6x1024x128xf32>
    %91 = tosa.add %89, %90 : (tensor<1x2x1x1024x128xf32>, tensor<1x2x6x1024x128xf32>) -> tensor<1x2x6x1024x128xf32>
    %92 = tosa.reshape %91 {new_shape = array<i64: 1, 12, 1024, 128>} : (tensor<1x2x6x1024x128xf32>) -> tensor<1x12x1024x128xf32>
    %93 = tosa.reshape %v_cache {new_shape = array<i64: 1, 2, 1, 1024, 128>} : (tensor<1x2x1024x128xf32>) -> tensor<1x2x1x1024x128xf32>
    %94 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x1024x128xf32>}> : () -> tensor<1x2x6x1024x128xf32>
    %95 = tosa.add %93, %94 : (tensor<1x2x1x1024x128xf32>, tensor<1x2x6x1024x128xf32>) -> tensor<1x2x6x1024x128xf32>
    %96 = tosa.reshape %95 {new_shape = array<i64: 1, 12, 1024, 128>} : (tensor<1x2x6x1024x128xf32>) -> tensor<1x12x1024x128xf32>


    %cst_21 = arith.constant 0.000000e+00 : f32
    %splat_22 = tensor.splat %cst_21 : tensor<1x1024xf32>
    %100 = tosa.reshape %mask {new_shape = array<i64: 1, 1024>} : (tensor<1x1x1x1024xf32>) -> tensor<1x1024xf32>
    %101 = tosa.add %splat_22, %100 : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %102 = tosa.reshape %q {new_shape = array<i64: 12, 1, 128>} : (tensor<1x12x1x128xf32>) -> tensor<12x1x128xf32>
    %103 = tosa.reshape %92 {new_shape = array<i64: 12, 1024, 128>} : (tensor<1x12x1024x128xf32>) -> tensor<12x1024x128xf32>
            // ===== Attention QK^T =====

    %cst_23 = arith.constant dense<0.000000e+00> : tensor<12x1x1024xf32>
    %104 = linalg.batch_matmul_transpose_b ins(%102, %103 : tensor<12x1x128xf32>, tensor<12x1024x128xf32>) outs(%cst_23 : tensor<12x1x1024xf32>) -> tensor<12x1x1024xf32>
    %cst_24 = arith.constant 0.0883883461 : f32
    %splat_25 = tensor.splat %cst_24 : tensor<12x1x1024xf32>
    %105 = tosa.mul %104, %splat_25 : (tensor<12x1x1024xf32>, tensor<12x1x1024xf32>) -> tensor<12x1x1024xf32>
    %106 = tosa.reshape %101 {new_shape = array<i64: 1, 1, 1024>} : (tensor<1x1024xf32>) -> tensor<1x1x1024xf32>
    %107 = tosa.add %105, %106 : (tensor<12x1x1024xf32>, tensor<1x1x1024xf32>) -> tensor<12x1x1024xf32>

    // ===== Attention Softmax =====

    %108 = tosa.reduce_max %107 {axis = 2 : i32} : (tensor<12x1x1024xf32>) -> tensor<12x1x1xf32>
    %109 = tosa.sub %107, %108 : (tensor<12x1x1024xf32>, tensor<12x1x1xf32>) -> tensor<12x1x1024xf32>
    %110 = math.exp %109 : tensor<12x1x1024xf32>
    %111 = tosa.reduce_sum %110 {axis = 2 : i32} : (tensor<12x1x1024xf32>) -> tensor<12x1x1xf32>
    %112 = tosa.log %111 : (tensor<12x1x1xf32>) -> tensor<12x1x1xf32>
    %113 = tosa.add %108, %112 : (tensor<12x1x1xf32>, tensor<12x1x1xf32>) -> tensor<12x1x1xf32>
    %114 = tosa.sub %107, %113 : (tensor<12x1x1024xf32>, tensor<12x1x1xf32>) -> tensor<12x1x1024xf32>
    %115 = math.exp %114 : tensor<12x1x1024xf32>

    // ===== Attention * V =====

    %116 = tosa.reshape %113 {new_shape = array<i64: 1, 12, 1>} : (tensor<12x1x1xf32>) -> tensor<1x12x1xf32>
    %117 = tosa.reshape %96 {new_shape = array<i64: 12, 1024, 128>} : (tensor<1x12x1024x128xf32>) -> tensor<12x1024x128xf32>
    %118 = tosa.matmul %115, %117 : (tensor<12x1x1024xf32>, tensor<12x1024x128xf32>) -> tensor<12x1x128xf32>
    %119 = tosa.reshape %118 {new_shape = array<i64: 1, 12, 1, 128>} : (tensor<12x1x128xf32>) -> tensor<1x12x1x128xf32>

    return %119 : tensor<1x12x1x128xf32>
  }

func.func @main() {

  %Q = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %k: index):
      %sum = arith.addi %b, %h : index
      %mix1 = arith.addi %sum, %i : index
      %mix2 = arith.addi %mix1, %k : index
      %c11 = arith.constant 11 : index
      %mod = arith.remui %mix2, %c11 : index
      %val = arith.index_cast %mod : index to i32
      %valf = arith.sitofp %val : i32 to f32
      tensor.yield %valf : f32
  } : tensor<1x12x1x128xf32>

  %K = tensor.generate {
    ^bb0(%b: index, %h: index, %k: index, %j: index):
      %sum = arith.addi %b, %h : index
      %mix1 = arith.addi %sum, %k : index
      %mix2 = arith.addi %mix1, %j : index
      %c17 = arith.constant 17 : index
      %mod = arith.remui %mix2, %c17 : index
      %val = arith.index_cast %mod : index to i32
      %valf = arith.sitofp %val : i32 to f32
      tensor.yield %valf : f32
  } : tensor<1x2x1024x128xf32>

  %V = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %k: index):
      %sum = arith.addi %b, %h : index
      %mix1 = arith.addi %sum, %i : index
      %mix2 = arith.addi %mix1, %k : index
      %c13 = arith.constant 13 : index
      %mod = arith.remui %mix2, %c13 : index
      %val = arith.index_cast %mod : index to i32
      %valf = arith.sitofp %val : i32 to f32
      tensor.yield %valf : f32
  } : tensor<1x2x1024x128xf32>


    // Mask: only allow j <= i positions, simulate causal mask
    %zero = arith.constant 0.0 : f32
    %neg  = arith.constant -1.0E+9 : f32

    %mask = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %j: index):
      %cond = arith.cmpi "slt", %i, %j : index
      %val = arith.select %cond, %neg, %zero : f32
      tensor.yield %val : f32
    } : tensor<1x1x1x1024xf32>

  %t_start = call @rtclock() : () -> f64

  %result_out = call @kernel(%Q, %K, %V, %mask) : (tensor<1x12x1x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  %tensor_unranked = tensor.cast %result_out : tensor<1x12x1x128xf32> to tensor<*xf32>
  // call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}
