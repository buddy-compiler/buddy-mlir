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
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @kernel(%144 : tensor<1x40x1536xf32>,%arg12 : tensor<1536xf32>) {
    %t_start = call @rtclock() : () -> f64
    
    %145 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_53 = arith.constant 2 : i32
    %146 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%144 : tensor<1x40x1536xf32>) outs(%145 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.fpowi %in, %c2_i32_53 : f32, i32
      linalg.yield %195 : f32
    } -> tensor<1x40x1536xf32>
    %147 = tosa.reduce_sum %146 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %148 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %149 = tosa.reciprocal %148 : (tensor<1xf32>) -> tensor<1xf32>
    %150 = tosa.reshape %149 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %151 = tosa.mul %150, %147 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %152 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %153 = tosa.add %151, %152 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %154 = tosa.rsqrt %153 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %155 = tosa.mul %144, %154 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %156 = tosa.reshape %arg12 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %157 = tosa.mul %156, %155 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
  
    %tensor_unranked = tensor.cast %157 : tensor<1x40x1536xf32> to tensor<*xf32>

      // Print results.
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}
  

  return
}
func.func @main() {

  %c0 = arith.constant dense<3.0> : tensor<1x40x1536xf32>
  %c1 = arith.constant dense <2.0> : tensor<1536xf32>


  call @kernel(%c0, %c1) : (tensor<1x40x1536xf32>, tensor<1536xf32>) -> ()

  return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

