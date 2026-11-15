// RUN: buddy-opt %s -lower-linalg-to-vir \
// RUN:     -lower-vir-to-vector='use-scalable=true rvv-vlen=128' | FileCheck %s
// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector='rvv-vlen=128' \
// RUN: | FileCheck %s

// A matmul of dynamic shape: M and K (the loop bounds) are taken before the
// vir.set_vl region, as lower-vir-to-vector lowers the ops of the region.
// CHECK-LABEL: func.func @dynamic
// CHECK: %[[M:.*]] = memref.dim %{{.*}}, %c0
// CHECK: %[[K:.*]] = memref.dim %{{.*}}, %c1
// CHECK: affine.for %{{.*}} = 0 to %[[M]]
// CHECK: affine.for %{{.*}} = 0 to %[[K]]
func.func @dynamic(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) {
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
                outs(%c : memref<?x?xf32>)
  return
}
