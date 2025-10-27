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
// RUN:     -matmul-parallel-vectorization-optimize \
// RUN:     -batchmatmul-optimize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:      -affine-loop-fusion \
// RUN:     -affine-parallelize \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-openmp \
// RUN:     -func-bufferize-dynamic-offset \
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
// RUN:     -reconcile-unrealized-casts  \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libomp%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func @kernel(%x: tensor<1x40x1536xf32>, %gamma: tensor<1536xf32>) -> (tensor<1x40x1536xf32>) {
  %eps = arith.constant 1.0e-5 : f32
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %dim_b = arith.constant 40 : index
  %dim_f = arith.constant 1536 : index
  %step = arith.constant 8 : index
  %vec_len = arith.constant 8.0 : f32

  %x_memref = bufferization.to_memref %x : tensor<1x40x1536xf32> to memref<1x40x1536xf32>
  %g_memref = bufferization.to_memref %gamma : tensor<1536xf32> to memref<1536xf32>
  %y_memref = memref.alloc() : memref<1x40x1536xf32>

  affine.for %b = 0 to %dim_b {
    affine.for %i = 0 to %dim_f step 8 iter_args(%M2_iter = %zero, %count_iter = %zero)-> (f32, f32) {
      %vx = vector.transfer_read %x_memref[%c0, %b, %i], %zero: memref<1x40x1536xf32>, vector<8xf32>
      %vg = vector.transfer_read %g_memref[%i], %zero: memref<1536xf32>, vector<8xf32>

      %vx_sq = arith.mulf %vx, %vx : vector<8xf32>
      %vx_sq_sum = vector.reduction <add>, %vx_sq : vector<8xf32> into f32

      // update count: count_new = count_iter + 8
      %count_add = arith.addf %count_iter, %vec_len : f32

      // update M2: M2_new = M2_iter + vx_sq_sum
      %M2_new = arith.addf %M2_iter, %vx_sq_sum : f32

      // update variance and standard deviation (actually RMS)
      %var = arith.divf %M2_new, %count_add : f32
      %var_eps = arith.addf %var, %eps : f32
      %inv_std = math.rsqrt %var_eps : f32
      %inv_std_vec = vector.splat %inv_std : vector<8xf32>
      %y_norm = arith.mulf %vx, %inv_std_vec : vector<8xf32>
      %y_scaled = arith.mulf %y_norm, %vg : vector<8xf32>

      // write result
      vector.transfer_write %y_scaled, %y_memref[%c0, %b, %i]: vector<8xf32>, memref<1x40x1536xf32>
      affine.yield %M2_new, %count_add : f32, f32
    }
  }

  %out = bufferization.to_tensor %y_memref restrict: memref<1x40x1536xf32> to tensor<1x40x1536xf32>
  return %out : tensor<1x40x1536xf32>
}

func.func @main() {

  %c0 = arith.constant dense<3.0> : tensor<1x40x1536xf32>
  %c1 = arith.constant dense <2.0> : tensor<1536xf32>

  %t_start = call @rtclock() : () -> f64
  %out = call @kernel(%c0, %c1) : (tensor<1x40x1536xf32>, tensor<1536xf32>) -> (tensor<1x40x1536xf32>)
  %t_end = call @rtclock() : () -> f64

  %tensor_unranked = tensor.cast %out : tensor<1x40x1536xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}


  return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

