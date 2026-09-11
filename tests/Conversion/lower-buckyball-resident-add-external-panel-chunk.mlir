// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s
// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s --check-prefix=SOURCE

// SOURCE-LABEL: func.func @external_add_cout40(
// SOURCE: memref.subview %arg1[0, 0, 0, 0] [1, 8, 1, 16]
// SOURCE: memref.subview %arg1[0, 8, 0, 0] [1, 8, 1, 16]
// SOURCE-NOT: memref.subview %arg1[
// SOURCE: return

// The first Conv is emitted before the second Conv's three-panel loop, whose
// MaxPool reads the same source bank. RHS DMA and the output channel guard keep
// eight lanes of panel 2; all three sum rows are copied through one bank chain.
// CHECK-LABEL: func.func @external_add_cout40(
// CHECK: memref.subview %arg1[
// CHECK: %[[FIRST_QUANT:[a-zA-Z0-9_]+]] = buckyball.bank_quant_i32_to_i8
// CHECK: %[[SRC_BANK:[a-zA-Z0-9_]+]] = buckyball.bank_maxpool %[[FIRST_QUANT]]
// CHECK: %[[LHS:[a-zA-Z0-9_]+]] = scf.for %[[PANEL:[a-zA-Z0-9_]+]] = %c0 to %c3 step %c1 iter_args(%[[DEST:[a-zA-Z0-9_]+]] = %{{[a-zA-Z0-9_]+}}) -> (i64)
// CHECK: buckyball.bank_maxpool %[[SRC_BANK]]
// CHECK: memref.subview %arg4[%[[PANEL]],
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = arith.index_cast %[[PANEL]] : index to i64
// CHECK: %[[QUANT:[a-zA-Z0-9_]+]] = buckyball.bank_quant_i32_to_i8
// CHECK: %[[COPIED:[a-zA-Z0-9_]+]] = buckyball.bank_maxpool %[[QUANT]] %[[DEST]] %c1_i64 %c0_i64 %[[OFFSET]] %c1_i64
// CHECK: scf.yield %[[COPIED]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: buckyball.bank_release %[[SRC_BANK]] : i64
// CHECK: %[[TAIL_SLICE:[a-zA-Z0-9_]+]] = memref.subview %arg8[0, 0, 0, 32] [1, 1, 1, 8]
// CHECK-NEXT: %[[TAIL_CAST:[a-zA-Z0-9_]+]] = memref.cast %[[TAIL_SLICE]]
// CHECK-NEXT: %[[RHS:[a-zA-Z0-9_]+]] = buckyball.bank_mvin_2d %[[TAIL_CAST]] %{{[a-zA-Z0-9_]+}} %c1_i64 %c40_i64 %c1_i64 %c2_i64 %c1_i64 %c8_i64
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = buckyball.bank_int8add %[[LHS]] %[[RHS]]
// CHECK-NEXT: %[[COPY0:[a-zA-Z0-9_]+]] = buckyball.bank_maxpool %[[SUM]] %{{[a-zA-Z0-9_]+}} %c1_i64 %c0_i64 %c0_i64 %c1_i64
// CHECK-NEXT: %[[COPY1:[a-zA-Z0-9_]+]] = buckyball.bank_maxpool %[[SUM]] %[[COPY0]] %c1_i64 %c1_i64 %c1_i64 %c1_i64
// CHECK-NEXT: %[[COPY2:[a-zA-Z0-9_]+]] = buckyball.bank_maxpool %[[SUM]] %[[COPY1]] %c1_i64 %c2_i64 %c2_i64 %c1_i64
// CHECK: buckyball.bank_release %[[SUM]] : i64
// CHECK: scf.for %[[OUT_PANEL:[a-zA-Z0-9_]+]] = %c0 to %c3 step %c1
// CHECK-NEXT: scf.for %[[OUT_LANE:[a-zA-Z0-9_]+]] = %c0 to %c16 step %c1
// CHECK-NEXT: %[[BASE:[a-zA-Z0-9_]+]] = arith.muli %[[OUT_PANEL]], %c16 : index
// CHECK-NEXT: %[[CHANNEL:[a-zA-Z0-9_]+]] = arith.addi %[[BASE]], %[[OUT_LANE]] : index
// CHECK-NEXT: %[[OUT_VALID:[a-zA-Z0-9_]+]] = arith.cmpi slt, %[[CHANNEL]], %c40 : index
// CHECK-NEXT: scf.if %[[OUT_VALID]]
// CHECK: memref.store {{.*}}, %arg9[%c0, %c0, %c0, %[[CHANNEL]]]

func.func @external_add_cout40(%input: memref<1x1x1x16xi8>, %first_weight: memref<1x16x16x16xi8>, %first_bias: memref<16xi32>, %first_scale: memref<16xf32>, %second_weight: memref<3x16x16x16xi8>, %second_bias: memref<40xi32>, %second_scale: memref<40xf32>, %lut: memref<1xi8>, %rhs: memref<1x1x1x40xi8>, %output: memref<1x1x1x40xi8>) {
  %first = memref.alloc() : memref<1x1x1x16xi8>
  %second = memref.alloc() : memref<1x1x1x40xi8>
  buckyball.mega_kernel %input %output : memref<1x1x1x16xi8> memref<1x1x1x40xi8> {
    buckyball.mega_conv2d %input %first_weight %first_bias %first_scale %lut %first {activation = 0 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x1x1x16xi8> memref<1x16x16x16xi8> memref<16xi32> memref<16xf32> memref<1xi8> memref<1x1x1x16xi8>
    buckyball.mega_conv2d %first %second_weight %second_bias %second_scale %lut %second {activation = 0 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x1x1x16xi8> memref<3x16x16x16xi8> memref<40xi32> memref<40xf32> memref<1xi8> memref<1x1x1x40xi8>
    buckyball.mega_int8_add %second %rhs %output {activation = 0 : i64, lhsScale = 1.0 : f32, rhsScale = 1.0 : f32, outputScale = 1.0 : f32} : memref<1x1x1x40xi8> memref<1x1x1x40xi8> memref<1x1x1x40xi8>
  }
  memref.dealloc %first : memref<1x1x1x16xi8>
  memref.dealloc %second : memref<1x1x1x40xi8>
  return
}
