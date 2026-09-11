// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s
// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -assign-physical-banks -lower-bank-ssa-to-intrinsics -canonicalize -cse | FileCheck %s --check-prefix=REDUCE

func.func @resident_mul_external_input(%input: memref<1x12x12x16xi8>, %output: memref<1x12x12x16xi8>) {
  %gate = memref.alloc() : memref<1x1x1x16xi8>
  buckyball.mega_kernel %input %output : memref<1x12x12x16xi8> memref<1x12x12x16xi8> {
    buckyball.mega_global_avg_pool %input %gate {inputScale = 1.0 : f32, outputScale = 1.0 : f32} : memref<1x12x12x16xi8> memref<1x1x1x16xi8>
    buckyball.mega_int8_mul %gate %input %output {activation = 0 : i64, lhsScale = 1.0 : f32, rhsScale = 1.0 : f32, outputScale = 1.0 : f32} : memref<1x1x1x16xi8> memref<1x12x12x16xi8> memref<1x12x12x16xi8>
  }
  memref.dealloc %gate : memref<1x1x1x16xi8>
  return
}

// CHECK-LABEL: func.func @resident_mul_external_input(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: memref<1x12x12x16xi8>
// CHECK: scf.for {{.*}} to %c12 step %c4
// CHECK: scf.for {{.*}} to %c12 step %c4
// CHECK: scf.for %[[Y:[a-zA-Z0-9_]+]] = %c0 to %c12 step %c2
// CHECK: scf.for %[[X:[a-zA-Z0-9_]+]] = %c0 to %c12 step %c2
// CHECK: %[[ACTIVATION:[a-zA-Z0-9_]+]] = scf.for %[[LY:[a-zA-Z0-9_]+]] = {{.*}} step %c1 iter_args
// CHECK: %[[ROW_RESULT:[a-zA-Z0-9_]+]] = scf.for %[[LX:[a-zA-Z0-9_]+]] = {{.*}} step %c8 iter_args
// CHECK: %[[GY:[a-zA-Z0-9_]+]] = arith.addi %[[Y]], %[[LY]] : index
// CHECK: %[[GX:[a-zA-Z0-9_]+]] = arith.addi %[[X]], %[[LX]] : index
// CHECK: %[[SLICE:[a-zA-Z0-9_]+]] = memref.subview %[[INPUT]][0, %[[GY]], %[[GX]], 0] [1, 1, %[[WIDTH:[a-zA-Z0-9_]+]], 16]
// CHECK: %[[CAST:[a-zA-Z0-9_]+]] = memref.cast %[[SLICE]]
// CHECK: %[[WIDTH64:[a-zA-Z0-9_]+]] = arith.index_cast %[[WIDTH]] : index to i64
// CHECK: %[[DMA:[a-zA-Z0-9_]+]] = buckyball.bank_mvin_2d %[[CAST]] %{{[a-zA-Z0-9_]+}} %c1_i64 %c16_i64 %c12_i64 %{{[a-zA-Z0-9_]+}} %[[WIDTH64]] %c16_i64
// CHECK-NEXT: scf.yield %[[DMA]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[ROW_RESULT]] : i64
// CHECK-NEXT: }
// CHECK: buckyball.bank_int8mul %{{[a-zA-Z0-9_]+}} %[[ACTIVATION]]
// CHECK-NOT: to %c12 step %c4
// CHECK: return

// REDUCE: scf.for {{.*}} to %[[EXTENT:c[0-9]+]] step %c4
// REDUCE: scf.for {{.*}} to %[[EXTENT]] step %c4
// REDUCE-NOT: to %[[EXTENT]] step %c4
