// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s

func.func @average_external_edge(%input: memref<1x14x14x16xi8>, %output: memref<1x1x1x16xi8>) {
  buckyball.mega_kernel %input %output : memref<1x14x14x16xi8> memref<1x1x1x16xi8> {
    buckyball.mega_global_avg_pool %input %output {inputScale = 1.0 : f32, outputScale = 1.0 : f32} : memref<1x14x14x16xi8> memref<1x1x1x16xi8>
  }
  return
}

// CHECK-LABEL: func.func @average_external_edge(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: memref<1x14x14x16xi8>
// CHECK: scf.for %[[Y:[a-zA-Z0-9_]+]] = %c0 to %c14 step %c4
// CHECK: scf.for %[[X:[a-zA-Z0-9_]+]] = %c0 to %c14 step %c4
// CHECK: %[[NEG_Y:[a-zA-Z0-9_]+]] = arith.subi %c0, %[[Y]] : index
// CHECK: %[[Y_BEGIN:[a-zA-Z0-9_]+]] = arith.maxsi %[[NEG_Y]], %c0 : index
// CHECK: %[[NEG_X:[a-zA-Z0-9_]+]] = arith.subi %c0, %[[X]] : index
// CHECK: %[[X_BEGIN:[a-zA-Z0-9_]+]] = arith.maxsi %[[NEG_X]], %c0 : index
// CHECK: %[[Y_REMAIN:[a-zA-Z0-9_]+]] = arith.subi %c14, %[[Y]] : index
// CHECK: %[[Y_END:[a-zA-Z0-9_]+]] = arith.minsi %[[Y_REMAIN]], %c4 : index
// CHECK: %[[X_REMAIN:[a-zA-Z0-9_]+]] = arith.subi %c14, %[[X]] : index
// CHECK: %[[X_END:[a-zA-Z0-9_]+]] = arith.minsi %[[X_REMAIN]], %c4 : index
// CHECK: scf.for %[[LOCAL_Y:[a-zA-Z0-9_]+]] = %[[Y_BEGIN]] to %[[Y_END]] step %c1
// CHECK: scf.for %[[LOCAL_X:[a-zA-Z0-9_]+]] = %[[X_BEGIN]] to %[[X_END]] step %c8
// CHECK: %[[GLOBAL_Y:[a-zA-Z0-9_]+]] = arith.addi %[[Y]], %[[LOCAL_Y]] : index
// CHECK: %[[GLOBAL_X:[a-zA-Z0-9_]+]] = arith.addi %[[X]], %[[LOCAL_X]] : index
// CHECK: %[[REMAIN:[a-zA-Z0-9_]+]] = arith.subi %[[X_END]], %[[LOCAL_X]] : index
// CHECK: %[[WIDTH:[a-zA-Z0-9_]+]] = arith.minsi %[[REMAIN]], %c8 : index
// CHECK: %[[SLICE:[a-zA-Z0-9_]+]] = memref.subview %[[INPUT]][0, %[[GLOBAL_Y]], %[[GLOBAL_X]], 0] [1, 1, %[[WIDTH]], 16]
// CHECK: %[[CAST:[a-zA-Z0-9_]+]] = memref.cast %[[SLICE]]
// CHECK: %[[ROW_Y:[a-zA-Z0-9_]+]] = arith.muli %[[LOCAL_Y]], %c4 : index
// CHECK: %[[ROW:[a-zA-Z0-9_]+]] = arith.addi %[[ROW_Y]], %[[LOCAL_X]] : index
// CHECK: %[[ROW64:[a-zA-Z0-9_]+]] = arith.index_cast %[[ROW]] : index to i64
// CHECK: %[[WIDTH64:[a-zA-Z0-9_]+]] = arith.index_cast %[[WIDTH]] : index to i64
// CHECK: buckyball.bank_mvin_2d %[[CAST]] %{{[a-zA-Z0-9_]+}} %c1_i64 %c16_i64 %c14_i64 %[[ROW64]] %[[WIDTH64]] %c16_i64
