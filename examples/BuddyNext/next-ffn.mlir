// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -one-shot-bufferize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -tensor-bufferize \
// RUN:     -buffer-deallocation \
// RUN:     -finalizing-bufferize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)>
#map7 = affine_map<(d0, d1) -> (0, d0, d1)>

func.func private @rtclock() -> f64

func.func @kernel_ffn(%arg0: tensor<1x40x4096xf32>, %arg9: tensor<4096xf32>, %arg10: tensor<11008x4096xf32>, %arg11: tensor<11008x4096xf32>, %arg12: tensor<4096x11008xf32>) {
  %t_start = call @rtclock() : () -> f64

  // FFN
  %138 = tosa.reshape %arg9 {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
  %139 = tosa.mul %138, %arg0 {shift = 0 : i8} : (tensor<1x1x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
  %140 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %141 = tosa.transpose %arg10, %140 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
  %142 = tosa.reshape %139 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_24 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
  %143 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%142, %141 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_24 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
  %144 = tosa.reshape %143 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
  %145 = tosa.sigmoid %144 : (tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
  %146 = tosa.mul %144, %145 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
  %147 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %148 = tosa.transpose %arg11, %147 : (tensor<11008x4096xf32>, tensor<2xi32>) -> tensor<4096x11008xf32>
  %149 = tosa.reshape %139 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
  %cst_25 = arith.constant dense<0.000000e+00> : tensor<40x11008xf32>
  %150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%149, %148 : tensor<40x4096xf32>, tensor<4096x11008xf32>) outs(%cst_25 : tensor<40x11008xf32>) -> tensor<40x11008xf32>
  %151 = tosa.reshape %150 {new_shape = array<i64: 1, 40, 11008>} : (tensor<40x11008xf32>) -> tensor<1x40x11008xf32>
  %152 = tosa.mul %146, %151 {shift = 0 : i8} : (tensor<1x40x11008xf32>, tensor<1x40x11008xf32>) -> tensor<1x40x11008xf32>
  %153 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %154 = tosa.transpose %arg12, %153 : (tensor<4096x11008xf32>, tensor<2xi32>) -> tensor<11008x4096xf32>
  %155 = tosa.reshape %152 {new_shape = array<i64: 40, 11008>} : (tensor<1x40x11008xf32>) -> tensor<40x11008xf32>
  %cst_26 = arith.constant dense<0.000000e+00> : tensor<40x4096xf32>
  %156 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%155, %154 : tensor<40x11008xf32>, tensor<11008x4096xf32>) outs(%cst_26 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %157 = tosa.reshape %156 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>
  %158 = tosa.add %arg0, %157 : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %158 : tensor<1x40x4096xf32> to tensor<*xf32>

  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  vector.print %time : f64

  return
}

func.func @main() {
  %input_tensor = arith.constant dense<3.0> : tensor<1x40x4096xf32>
  %weight1 = arith.constant dense<1.0> : tensor<4096xf32>
  %weight2 = arith.constant dense<1.0> : tensor<11008x4096xf32>
  %weight3 = arith.constant dense<2.0> : tensor<11008x4096xf32>
  %weight4 = arith.constant dense<1.0> : tensor<4096x11008xf32>

  call @kernel_ffn(%input_tensor, %weight1, %weight2, %weight3, %weight4) : (tensor<1x40x4096xf32>, tensor<4096xf32>, tensor<11008x4096xf32>, tensor<11008x4096xf32>, tensor<4096x11008xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)