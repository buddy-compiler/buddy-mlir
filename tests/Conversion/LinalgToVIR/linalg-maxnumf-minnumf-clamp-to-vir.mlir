// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

func.func @clamp_with_maxnumf_minnumf(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  linalg.generic
      { indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"] }
      ins(%arg0 : memref<?xf32>)
      outs(%arg1 : memref<?xf32>) {
    ^bb0(%a: f32, %b: f32):
      %lo = arith.constant -1.0 : f32
      %hi = arith.constant 1.0 : f32
      %m0 = arith.maxnumf %a, %lo : f32
      %m1 = arith.minnumf %m0, %hi : f32
      linalg.yield %m1 : f32
  }
  return
}

// CHECK-LABEL: func.func @clamp_with_maxnumf_minnumf
// CHECK: %[[VL:.*]] = memref.dim %arg1, %{{.*}} : memref<?xf32>
// CHECK: vir.set_vl %[[VL]] : index {
// CHECK:   %[[A:.*]] = vir.load %{{.*}}[] : memref<?xf32, strided<[1]>> -> !vir.vec<?xf32>
// CHECK:   %[[LO:.*]] = vir.broadcast %{{.*}} : f32 -> !vir.vec<?xf32>
// CHECK:   %[[HI:.*]] = vir.broadcast %{{.*}} : f32 -> !vir.vec<?xf32>
// CHECK:   %[[M0:.*]] = arith.maxnumf %[[A]], %[[LO]] : !vir.vec<?xf32>
// CHECK:   %[[M1:.*]] = arith.minnumf %[[M0]], %[[HI]] : !vir.vec<?xf32>
// CHECK:   vir.store %[[M1]], %{{.*}}[] : !vir.vec<?xf32> -> memref<?xf32, strided<[1]>>
// CHECK: }
// CHECK-NOT: linalg.generic
