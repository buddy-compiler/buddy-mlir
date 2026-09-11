// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s

// Cin=40 requires panels with 16, 16, and 8 lanes. The accumulator crosses
// both loops; only global channels 0 and 39 initialize/finalize the sum.
// CHECK-LABEL: func.func @external_cin40(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: memref<1x1x1x40xi8>
// CHECK: %[[BIAS:[a-zA-Z0-9_]+]] = buckyball.bank_smatmul_bias
// CHECK-NEXT: buckyball.bank_release %[[BIAS]] : i64
// CHECK-NEXT: %[[P0:[a-zA-Z0-9_]+]] = buckyball.bank_alloc
// CHECK-NEXT: %[[W0:[a-zA-Z0-9_]+]] = buckyball.bank_alloc
// CHECK-NEXT: %[[R0:[a-zA-Z0-9_]+]] = buckyball.bank_alloc
// CHECK-NEXT: %[[PANELS:[a-zA-Z0-9_]+]]:3 = scf.for %[[PANEL:[a-zA-Z0-9_]+]] = %c0 to %c3 step %c1 iter_args(%[[PP:[a-zA-Z0-9_]+]] = %[[P0]], %[[PW:[a-zA-Z0-9_]+]] = %[[W0]], %[[PR:[a-zA-Z0-9_]+]] = %[[R0]]) -> (i64, i64, i64)
// CHECK: %[[BEGIN:[a-zA-Z0-9_]+]] = arith.muli %[[PANEL]], %c16 : index
// CHECK-NEXT: %[[REMAIN:[a-zA-Z0-9_]+]] = arith.subi %c40, %[[BEGIN]] : index
// CHECK-NEXT: %[[LANES:[a-zA-Z0-9_]+]] = arith.minsi %[[REMAIN]], %c16 : index
// CHECK-NEXT: %[[BYTES:[a-zA-Z0-9_]+]] = arith.index_cast %[[LANES]] : index to i64
// CHECK-NEXT: %[[SLICE:[a-zA-Z0-9_]+]] = memref.subview %[[INPUT]][0, 0, 0, %[[BEGIN]]] [1, 1, 1, %[[LANES]]]
// CHECK-NEXT: %[[CAST:[a-zA-Z0-9_]+]] = memref.cast %[[SLICE]]
// CHECK-NEXT: %[[SOURCE:[a-zA-Z0-9_]+]] = buckyball.bank_mvin_2d %[[CAST]] %{{[a-zA-Z0-9_]+}} %c1_i64 %c40_i64 %c1_i64 %c0_i64 %c1_i64 %[[BYTES]]
// CHECK: %[[NEXT_PANEL:[a-zA-Z0-9_]+]] = arith.addi %[[PANEL]], %c1 : index
// CHECK-NEXT: %[[PAD_END:[a-zA-Z0-9_]+]] = arith.muli %[[NEXT_PANEL]], %c16 : index
// CHECK-NEXT: %[[END:[a-zA-Z0-9_]+]] = arith.minui %[[PAD_END]], %c40 : index
// CHECK-NEXT: %[[CHANNELS:[a-zA-Z0-9_]+]]:3 = scf.for %[[CHANNEL:[a-zA-Z0-9_]+]] = %[[BEGIN]] to %[[END]] step %c1 iter_args(%[[CP:[a-zA-Z0-9_]+]] = %[[PP]], %[[CW:[a-zA-Z0-9_]+]] = %[[PW]], %[[CR:[a-zA-Z0-9_]+]] = %[[PR]]) -> (i64, i64, i64)
// CHECK: %[[PATCH:[a-zA-Z0-9_]+]] = buckyball.bank_im2col %[[SOURCE]] %[[CP]]
// CHECK: %[[WEIGHT:[a-zA-Z0-9_]+]] = buckyball.bank_mvin %{{[a-zA-Z0-9_]+}} %[[CW]]
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[CHANNEL]], %c0 : index
// CHECK-NEXT: %[[LAST:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[CHANNEL]], %c39 : index
// CHECK-NEXT: %[[SUM:[a-zA-Z0-9_]+]] = buckyball.bank_smatmul %[[PATCH]] %[[WEIGHT]] %[[CR]] %{{[a-zA-Z0-9_]+}} %[[FIRST]] %[[LAST]]
// CHECK-NEXT: scf.yield %[[PATCH]], %[[WEIGHT]], %[[SUM]] : i64, i64, i64
// CHECK-NEXT: }
// CHECK-NEXT: buckyball.bank_release %[[SOURCE]] : i64
// CHECK-NEXT: scf.yield %[[CHANNELS]]#0, %[[CHANNELS]]#1, %[[CHANNELS]]#2 : i64, i64, i64
// CHECK-NEXT: }
// CHECK-NEXT: buckyball.bank_release %[[PANELS]]#0 : i64
// CHECK-NEXT: buckyball.bank_release %[[PANELS]]#1 : i64
// CHECK: buckyball.bank_quant_i32_to_i8 %[[PANELS]]#2

func.func @external_cin40(%input: memref<1x1x1x40xi8>, %weight: memref<1x40x16x16xi8>, %bias: memref<16xi32>, %scale: memref<16xf32>, %lut: memref<1xi8>, %output: memref<1x1x1x16xi8>) {
  buckyball.mega_kernel %input %output : memref<1x1x1x40xi8> memref<1x1x1x16xi8> {
    buckyball.mega_conv2d %input %weight %bias %scale %lut %output {activation = 0 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x1x1x40xi8> memref<1x40x16x16xi8> memref<16xi32> memref<16xf32> memref<1xi8> memref<1x1x1x16xi8>
  }
  return
}
