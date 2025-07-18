// RUN: buddy-opt %s \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
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
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @kernel(%arg0: tensor<40x4096xf32>, %arg1: tensor<4096x4096xf32>)
    -> tensor<1x40x4096xf32> {
  %cst = arith.constant dense<0.0> : tensor<40x4096xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<40x4096xf32>, tensor<4096x4096xf32>)
                     outs(%cst : tensor<40x4096xf32>) -> tensor<40x4096xf32>
  %expanded = tensor.expand_shape %0 [[0, 1], [2]] output_shape [1, 40, 4096] : tensor<40x4096xf32>
      into tensor<1x40x4096xf32>
  %1 = tensor.empty() : tensor<1x40x4096xf32>
  %c2_i32 = arith.constant 2 : i32
  %2 = linalg.generic
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<1x40x4096xf32>) outs(%1 : tensor<1x40x4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = math.fpowi %in, %c2_i32 : f32, i32
    linalg.yield %3 : f32
  } -> tensor<1x40x4096xf32>
  return %2 : tensor<1x40x4096xf32>
}

func.func @main() {

  %c0 = arith.constant dense<3.0> : tensor<40x4096xf32>
  %c1 = arith.constant dense <2.0> : tensor<4096x4096xf32>

  %res = call @kernel(%c0, %c1) : (tensor<40x4096xf32>, tensor<4096x4096xf32>) -> (tensor<1x40x4096xf32>)

  %tensor_unranked = tensor.cast %res : tensor<1x40x4096xf32> to tensor<*xf32>
  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 4096] strides = [163840, 4096, 1] data =
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
