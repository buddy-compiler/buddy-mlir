// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -convert-elementwise-to-linalg \
// RUN:     -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
// RUN:     -expand-strided-metadata \
// RUN:     -ownership-based-buffer-deallocation \
// RUN:     -buffer-deallocation-simplification \
// RUN:     -bufferization-lower-deallocations \
// RUN:     -convert-bufferization-to-memref \
// RUN:     -matmul-parallel-vectorization-optimize \
// RUN:     -batchmatmul-optimize \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
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
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libomp%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @kernel(%arg0: tensor<1x1024x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x1024x1536xf32> {
    %t_start = call @rtclock() : () -> f64

    %eps = arith.constant 9.99999997E-7 : f32
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %c1536 = arith.constant 1536 : index
    %dim = arith.constant 1.536000e+03 : f32

    %x_memref = bufferization.to_memref %arg0 : tensor<1x1024x1536xf32> to memref<1x1024x1536xf32>
    %g_memref = bufferization.to_memref %arg1 : tensor<1536xf32> to memref<1536xf32>
    %y_memref = memref.alloc() : memref<1x1024x1536xf32>

    scf.for %b = %c0 to %c1024 step %c1 {
        %acc = scf.for %i = %c0 to %c1536 step %c16 iter_args(%acc_iter = %zero) -> (f32) {
          %x_vec = vector.load %x_memref[%c0, %b, %i] : memref<1x1024x1536xf32>, vector<16xf32>
          %x_sq_vec = arith.mulf %x_vec, %x_vec : vector<16xf32>
          %partial = vector.reduction <add>, %x_sq_vec : vector<16xf32> into f32
          %acc_new = arith.addf %acc_iter, %partial : f32
          scf.yield %acc_new : f32
        }
        %mean = arith.divf %acc, %dim : f32
        %m_eps = arith.addf %mean, %eps : f32
        %inv_rms = math.rsqrt %m_eps : f32
        %inv_vec = vector.splat %inv_rms : vector<16xf32>
        scf.for %i = %c0 to %c1536 step %c16 {
          %x_vec = vector.load %x_memref[%c0, %b, %i] : memref<1x1024x1536xf32>, vector<16xf32>
          %g_vec = vector.load %g_memref[%i] : memref<1536xf32>, vector<16xf32>
          %x_norm_vec = arith.mulf %x_vec, %inv_vec : vector<16xf32>
          %y_vec = arith.mulf %x_norm_vec, %g_vec : vector<16xf32>
          vector.store %y_vec, %y_memref[%c0, %b, %i] : memref<1x1024x1536xf32>, vector<16xf32>
        }
    }

    %out = bufferization.to_tensor %y_memref restrict : memref<1x1024x1536xf32> to tensor<1x1024x1536xf32>

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64

    // Print timings.
    vector.print %time : f64
    // CHECK: {{[0-9]+\.[0-9]+}}

    return %out : tensor<1x1024x1536xf32>
}

func.func @main() {

  %cst_3 = arith.constant 3.0 : f32
  %empty_0 = tensor.empty() : tensor<1x1024x1536xf32>
  %c0 = linalg.fill ins(%cst_3 : f32) outs(%empty_0 : tensor<1x1024x1536xf32>) -> tensor<1x1024x1536xf32>

  %cst_2 = arith.constant 2.0 : f32
  %empty_1 = tensor.empty() : tensor<1536xf32>
  %c1 = linalg.fill ins(%cst_2 : f32) outs(%empty_1 : tensor<1536xf32>) -> tensor<1536xf32>

  %res = call @kernel(%c0, %c1) : (tensor<1x1024x1536xf32>, tensor<1536xf32>) -> tensor<1x1024x1536xf32>

  %tensor_unranked = tensor.cast %res : tensor<1x1024x1536xf32> to tensor<*xf32>
  // Print results.
//   call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}
