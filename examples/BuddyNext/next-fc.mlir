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

func.func private @rtclock() -> f64

func.func @kernel_fc_layer(%arg0 : tensor<1x40x4096xf32>, %arg1 : tensor<4096x4096xf32>, %arg2 : tensor<4096x4096xf32>, %arg3 : tensor<1x40x4096xf32>) {
%t_start = call @rtclock() : () -> f64

%cst_0 = arith.constant dense<0.0> : tensor<40x4096xf32>
%cst_1 = arith.constant dense<0.0> : tensor<40x4096xf32>

%41 = tosa.mul %arg0, %arg3 {shift = 0 : i8} : (tensor<1x40x4096xf32>, tensor<1x40x4096xf32>) -> tensor<1x40x4096xf32>
%42 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
%43 = tosa.transpose %arg1, %42 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
%44 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
%45 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%44, %43 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_0 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
%46 = tosa.reshape %45 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

%47 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
%48 = tosa.transpose %arg2, %47 : (tensor<4096x4096xf32>, tensor<2xi32>) -> tensor<4096x4096xf32>
%49 = tosa.reshape %41 {new_shape = array<i64: 40, 4096>} : (tensor<1x40x4096xf32>) -> tensor<40x4096xf32>
%50 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%49, %48 : tensor<40x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1 : tensor<40x4096xf32>) -> tensor<40x4096xf32>
%51 = tosa.reshape %50 {new_shape = array<i64: 1, 40, 4096>} : (tensor<40x4096xf32>) -> tensor<1x40x4096xf32>

%t_end = call @rtclock() : () -> f64
%time = arith.subf %t_end, %t_start : f64

%tensor_unranked = tensor.cast %51 : tensor<1x40x4096xf32> to tensor<*xf32>

call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
vector.print %time : f64

return
}

func.func @main() {
%input_tensor_1 = arith.constant dense<3.0> : tensor<1x40x4096xf32>
%input_tensor_2 = arith.constant dense<2.0> : tensor<4096x4096xf32>
%input_tensor_3 = arith.constant dense<1.0> : tensor<4096x4096xf32>
%input_tensor_4 = arith.constant dense<4.0> : tensor<1x40x4096xf32>

call @kernel_fc_layer(%input_tensor_1, %input_tensor_2, %input_tensor_3, %input_tensor_4) : (tensor<1x40x4096xf32>, tensor<4096x4096xf32>, tensor<4096x4096xf32>, tensor<1x40x4096xf32>) -> ()

return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)