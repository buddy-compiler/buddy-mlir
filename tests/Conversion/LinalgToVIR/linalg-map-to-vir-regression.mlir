// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s

func.func @map_scalar_defined_above(%out: memref<4xi32>, %seed: i32) {
  linalg.map outs(%out : memref<4xi32>)
    () {
      linalg.yield %seed : i32
    }
  return
}

// CHECK-LABEL: func.func @map_scalar_defined_above
// CHECK: vir.set_vl %[[VL:.*]] : index {
// CHECK:   %[[TOUT:.*]] = memref.transpose %arg0 (d0) -> (d0) : memref<4xi32> to memref<4xi32, strided<[1]>>
// CHECK:   %[[V:.*]] = vir.broadcast %arg1 : i32 -> !vir.vec<?xi32>
// CHECK:   vir.store %[[V]], %[[TOUT]][]
