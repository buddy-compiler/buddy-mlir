// RUN: buddy-opt -sigmoid-vectorize-manual %s | FileCheck %s

// CHECK: module {
// CHECK:  func.func @sigmoid_test(%arg0: tensor<1x8x8xf32>) {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    %c0_0 = arith.constant 0 : index
// CHECK-NEXT:    %dim = tensor.dim %arg0, %c0_0 : tensor<1x8x8xf32>
// CHECK-NEXT:    %c1_1 = arith.constant 1 : index
// CHECK-NEXT:    %dim_2 = tensor.dim %arg0, %c1_1 : tensor<1x8x8xf32>
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %dim_3 = tensor.dim %arg0, %c2 : tensor<1x8x8xf32>
// CHECK-NEXT:    %alloc = memref.alloc() : memref<1x8x8xf32>
// CHECK-NEXT:    scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%dim, %dim_2) step (%c1, %c1) {
// CHECK-NEXT:      %cst_4 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %0 = vector.transfer_read %arg0[%arg1, %arg2, %c0], %cst_4 : tensor<1x8x8xf32>, vector<8xf32>
// CHECK-NEXT:      %1 = arith.negf %0 : vector<8xf32>
// CHECK-NEXT:      %2 = math.exp %1 : vector<8xf32>
// CHECK-NEXT:      %3 = vector.broadcast %cst : f32 to vector<8xf32>
// CHECK-NEXT:      %4 = arith.addf %2, %3 : vector<8xf32>
// CHECK-NEXT:      %5 = arith.divf %3, %4 : vector<8xf32>
// CHECK-NEXT:      vector.transfer_write %5, %alloc[%arg1, %arg2, %c0] {in_bounds = [true]} : vector<8xf32>, memref<1x8x8xf32>
// CHECK-NEXT:      scf.reduce 
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT: }

func.func @sigmoid_test(%arg0: tensor<1x8x8xf32>) {
  %sigmoid = tosa.sigmoid %arg0:(tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
  return
}

