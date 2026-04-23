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


func.func private @rtclock() -> f64
func.func @kernel(
  %10830: tensor<1x1536xf32>,
  %arg606: tensor<262144x1536xf32>
) -> tensor<1x1x262144xf32> {
    // ===== LM Head Timing Start =====
    %t_lm_head_start = call @rtclock() : () -> f64
    %cst_1441 = arith.constant dense<0.000000e+00> : tensor<1x262144xf32>
    %10831 = linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_signed>} ins(%10830, %arg606 : tensor<1x1536xf32>, tensor<262144x1536xf32>) outs(%cst_1441 : tensor<1x262144xf32>) -> tensor<1x262144xf32>


    %10832 = tosa.const_shape  {values = dense<[1, 1, 262144]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %10833 = tosa.reshape %10831, %10832 : (tensor<1x262144xf32>, !tosa.shape<3>) -> tensor<1x1x262144xf32>
    %10834 = "tosa.const"() <{values = dense<3.000000e+01> : tensor<1x1x262144xf32>}> : () -> tensor<1x1x262144xf32>
    %10835 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %10836 = tosa.reciprocal %10834 : (tensor<1x1x262144xf32>) -> tensor<1x1x262144xf32>
    %10837 = tosa.mul %10833, %10836, %10835 : (tensor<1x1x262144xf32>, tensor<1x1x262144xf32>, tensor<1xi8>) -> tensor<1x1x262144xf32>
    %10838 = tosa.tanh %10837 : (tensor<1x1x262144xf32>) -> tensor<1x1x262144xf32>
    %cst_1442 = arith.constant dense<3.000000e+01> : tensor<1xf32>
    %10839 = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
    %10840 = tosa.reshape %cst_1442, %10839 : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
    %10841 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %10842 = tosa.mul %10838, %10840, %10841 : (tensor<1x1x262144xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x262144xf32>

        // ===== LM Head Timing End =====
    %t_lm_head_end = call @rtclock() : () -> f64
    %t_elapsed_lm_head = arith.subf %t_lm_head_end, %t_lm_head_start : f64
    vector.print %t_elapsed_lm_head : f64

 return %10842 : tensor<1x1x262144xf32>
}

func.func @main() {
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %zero = arith.constant 0.0 : f32
  %neg = arith.constant -1.0 : f32
  %c768 = arith.constant 768 : index

  %hidden = tensor.generate {
    ^bb0(%b: index, %h: index):
      %cond = arith.cmpi slt, %h, %c768 : index
      %val = arith.select %cond, %two, %one : f32
      tensor.yield %val : f32
  } : tensor<1x1536xf32>

  %lm_weight = tensor.generate {
    ^bb0(%i: index, %j: index):
      %cond = arith.cmpi slt, %i, %j : index
      %val = arith.select %cond, %neg, %zero : f32
      tensor.yield %val : f32
  } : tensor<262144x1536xf32>

    %t_start = call @rtclock() : () -> f64

  %result_out = call @kernel(%hidden, %lm_weight) : (tensor<1x1536xf32>, tensor<262144x1536xf32>) -> tensor<1x1x262144xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %result_out : tensor<1x1x262144xf32> to tensor<*xf32>
  // call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}
  return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
