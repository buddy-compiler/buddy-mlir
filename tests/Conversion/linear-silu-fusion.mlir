// RUN: buddy-opt %s -linear-silu-fusion -linear-silu-lower | FileCheck %s
// CHECK-LABEL: module {
// CHECK:   func.func @linear_silu(%arg0: tensor<1x1x784xf32>, %arg1: tensor<32x784xf32>, %arg2: tensor<32xf32>) -> tensor<1x1x32xf32> {
// CHECK-NEXT:     %cst = arith.constant dense<1.000000e+00> : vector<8xf32>
// CHECK-NEXT:     %cst_0 = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-NEXT:     %0 = tensor.empty() : tensor<784x32xf32>
// CHECK-NEXT:     %transposed = linalg.transpose ins(%arg1 : tensor<32x784xf32>) outs(%0 : tensor<784x32xf32>) permutation = [1, 0]
// CHECK-NEXT:     %expanded = tensor.expand_shape %transposed 
// CHECK-NEXT:     %expanded_1 = tensor.expand_shape %arg2 
// CHECK-NEXT:     %expanded_2 = tensor.expand_shape %expanded_1 
// CHECK-NEXT:     %1 = tensor.empty() : tensor<1x1x32xf32>
// CHECK-NEXT:     %2 = bufferization.to_memref %arg0 : tensor<1x1x784xf32> to memref<1x1x784xf32>
// CHECK-NEXT:     %3 = bufferization.to_memref %expanded : tensor<1x784x32xf32> to memref<1x784x32xf32>
// CHECK-NEXT:     %4 = bufferization.to_memref %expanded_2 : tensor<1x1x32xf32> to memref<1x1x32xf32>
// CHECK-NEXT:     %5 = bufferization.to_memref %1 : tensor<1x1x32xf32> to memref<1x1x32xf32>
// CHECK-NEXT:     %alloc = memref.alloc() : memref<vector<8xf32>>
// CHECK-NEXT:     affine.for %arg3 = 0 to 1 {
// CHECK-NEXT:       affine.for %arg4 = 0 to 1 {
// CHECK-NEXT:         affine.for %arg5 = 0 to 32 step 8 {
// CHECK-NEXT:           memref.store %cst_0, %alloc[] : memref<vector<8xf32>>
// CHECK-NEXT:           affine.for %arg6 = 0 to 784 {
// CHECK-NEXT:             %15 = affine.load %2[%arg3, %arg4, %arg6] : memref<1x1x784xf32>
// CHECK-NEXT:             %16 = vector.splat %15 : vector<8xf32>
// CHECK-NEXT:             %17 = vector.load %3[%arg3, %arg6, %arg5] : memref<1x784x32xf32>, vector<8xf32>
// CHECK-NEXT:             %18 = arith.mulf %16, %17 : vector<8xf32>
// CHECK-NEXT:             %19 = memref.load %alloc[] : memref<vector<8xf32>>
// CHECK-NEXT:             %20 = arith.addf %19, %18 : vector<8xf32>
// CHECK-NEXT:             memref.store %20, %alloc[] : memref<vector<8xf32>>
// CHECK-NEXT:           }
// CHECK-NEXT:           %7 = memref.load %alloc[] : memref<vector<8xf32>>
// CHECK-NEXT:           %8 = vector.load %4[%arg3, %arg4, %arg5] : memref<1x1x32xf32>, vector<8xf32>
// CHECK-NEXT:           %9 = arith.addf %7, %8 : vector<8xf32>
// CHECK-NEXT:           %10 = arith.negf %9 : vector<8xf32>
// CHECK-NEXT:           %11 = math.exp %10 : vector<8xf32>
// CHECK-NEXT:           %12 = arith.addf %11, %cst : vector<8xf32>
// CHECK-NEXT:           %13 = arith.divf %cst, %12 : vector<8xf32>
// CHECK-NEXT:           %14 = arith.mulf %9, %13 : vector<8xf32>
// CHECK-NEXT:           vector.store %14, %5[%arg3, %arg4, %arg5] : memref<1x1x32xf32>, vector<8xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     %6 = bufferization.to_tensor %5 restrict : memref<1x1x32xf32> to tensor<1x1x32xf32>
// CHECK-NEXT:     return %6 : tensor<1x1x32xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }


#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @linear_silu(%arg0: tensor<1x1x784xf32>, %arg1: tensor<32x784xf32>, %arg2: tensor<32xf32>) -> tensor<1x1x32xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x1x784xf32> into tensor<1x784xf32>
    %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
    %0 = tensor.empty() : tensor<784x32xf32>
    %transposed = linalg.transpose ins(%arg1 : tensor<32x784xf32>) outs(%0 : tensor<784x32xf32>) permutation = [1, 0] 
    %expanded = tensor.expand_shape %transposed [[0, 1], [2]] output_shape [1, 784, 32] : tensor<784x32xf32> into tensor<1x784x32xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<1x1x32xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
    %3 = linalg.batch_matmul ins(%arg0, %expanded : tensor<1x1x784xf32>, tensor<1x784x32xf32>) outs(%2 : tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
    %collapsed_1 = tensor.collapse_shape %3 [[0, 1], [2]] : tensor<1x1x32xf32> into tensor<1x32xf32>
    %expanded_2 = tensor.expand_shape %arg2 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
    %4 = tensor.empty() : tensor<1x32xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_2, %collapsed_1 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%4 : tensor<1x32xf32>) {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %26 = arith.addf %in, %in_20 : f32
      linalg.yield %26 : f32
    } -> tensor<1x32xf32>
    %expanded_3 = tensor.expand_shape %5 [[0, 1], [2]] output_shape [1, 1, 32] : tensor<1x32xf32> into tensor<1x1x32xf32>
    %6 = tensor.empty() : tensor<1x1x32xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<1x1x32xf32>) outs(%6 : tensor<1x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_20 = arith.constant 1.000000e+00 : f32
      %26 = arith.negf %in : f32
      %27 = math.exp %26 : f32
      %28 = arith.addf %27, %cst_20 : f32
      %29 = arith.divf %cst_20, %28 : f32
      linalg.yield %29 : f32
    } -> tensor<1x1x32xf32>
    %8 = tensor.empty() : tensor<1x1x32xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_3, %7 : tensor<1x1x32xf32>, tensor<1x1x32xf32>) outs(%8 : tensor<1x1x32xf32>) {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %26 = arith.mulf %in, %in_20 : f32
      linalg.yield %26 : f32
    } -> tensor<1x1x32xf32>
    return %9 : tensor<1x1x32xf32>
    }
}