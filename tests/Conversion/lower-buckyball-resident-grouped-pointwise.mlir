// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -canonicalize -cse | FileCheck %s
// RUN: buddy-opt %s --target=pebble -lower-buckyball-to-bank-ssa -assign-physical-banks -lower-bank-ssa-to-intrinsics -lower-buckyball -canonicalize -cse -llvm-request-c-wrappers -expand-strided-metadata -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -buffer-deallocation-simplification -bufferization-lower-deallocations -convert-math-to-llvm -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=LLVM

// CHECK-LABEL: func.func @grouped_cin32_cout40(
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = memref.alloc() : memref<64x16xi8>
// CHECK: linalg.fill ins(%c0_i8 : i8) outs(%[[ZERO]] : memref<64x16xi8>)
// CHECK: %{{.*}}, %[[STRIDES:[a-zA-Z0-9_]+]]:4 = memref.extract_strided_metadata %arg1
// CHECK: %[[LANES:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[STRIDES]]#3, %c1 : index
// CHECK-NEXT: %[[LOW:[a-zA-Z0-9_]+]] = arith.cmpi sge, %[[STRIDES]]#1, %c16 : index
// CHECK-NEXT: %[[HIGH:[a-zA-Z0-9_]+]] = arith.cmpi sle, %[[STRIDES]]#1, %c1016 : index
// CHECK-NEXT: %[[REM:[a-zA-Z0-9_]+]] = arith.remsi %[[STRIDES]]#1, %c8 : index
// CHECK-NEXT: %[[ALIGNED:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[REM]], %c0 : index
// CHECK-NEXT: %[[A:[a-zA-Z0-9_]+]] = arith.andi %[[LANES]], %[[LOW]] : i1
// CHECK-NEXT: %[[B:[a-zA-Z0-9_]+]] = arith.andi %[[HIGH]], %[[ALIGNED]] : i1
// CHECK-NEXT: %[[VALID:[a-zA-Z0-9_]+]] = arith.andi %[[A]], %[[B]] : i1
// CHECK-NEXT: cf.assert %[[VALID]], "1x1 Conv weight requires contiguous lanes and a channel stride between 16 and 1016 bytes divisible by 8"
// CHECK-NEXT: %[[STRIDE64:[a-zA-Z0-9_]+]] = arith.index_cast %[[STRIDES]]#1 : index to i64
// CHECK: scf.for %[[OUT:[a-zA-Z0-9_]+]] = %c0 to %c3 step %c1 iter_args
// CHECK: buckyball.bank_smatmul_bias
// CHECK: %[[P0:[a-zA-Z0-9_]+]] = buckyball.bank_alloc
// CHECK-NEXT: %[[W0:[a-zA-Z0-9_]+]] = buckyball.bank_alloc
// CHECK-NEXT: %[[R0:[a-zA-Z0-9_]+]] = buckyball.bank_alloc
// CHECK-NEXT: %[[PZERO:[a-zA-Z0-9_]+]] = buckyball.bank_mvin %[[ZERO]] %[[P0]] %c16_i64 %c1_i64
// CHECK-NEXT: %[[PANELS:[a-zA-Z0-9_]+]]:3 = scf.for %[[PANEL:[a-zA-Z0-9_]+]] = %c0 to %c2 step %c1 iter_args(%[[PP:[a-zA-Z0-9_]+]] = %[[PZERO]], %[[PW:[a-zA-Z0-9_]+]] = %[[W0]], %[[PR:[a-zA-Z0-9_]+]] = %[[R0]]) -> (i64, i64, i64)
// CHECK: %[[BEGIN:[a-zA-Z0-9_]+]] = arith.muli %[[PANEL]], %c16 : index
// CHECK: %[[SRC:[a-zA-Z0-9_]+]] = buckyball.bank_mvin_2d
// CHECK: %[[CHLOOP:[a-zA-Z0-9_]+]]:3 = scf.for %[[CH:[a-zA-Z0-9_]+]] = %[[BEGIN]] to %{{[a-zA-Z0-9_]+}} step %c16 iter_args(%[[CP:[a-zA-Z0-9_]+]] = %[[PP]], %[[CW:[a-zA-Z0-9_]+]] = %[[PW]], %[[CR:[a-zA-Z0-9_]+]] = %[[PR]]) -> (i64, i64, i64)
// CHECK: %[[PATCH:[a-zA-Z0-9_]+]] = buckyball.bank_maxpool %[[SRC]] %[[CP]] %c1_i64 %{{[a-zA-Z0-9_]+}} %c0_i64 %c1_i64
// CHECK: %[[HALF0:[a-zA-Z0-9_]+]] = memref.subview %arg1[%[[OUT]], %[[CH]], 0, 0] [1, 8, 1, 16]
// CHECK-NEXT: %[[W1:[a-zA-Z0-9_]+]] = buckyball.bank_mvin_2d %[[HALF0]] %[[CW]] %c1_i64 %[[STRIDE64]] %c8_i64 %c0_i64 %c8_i64 %c16_i64
// CHECK-NEXT: %[[CH8:[a-zA-Z0-9_]+]] = arith.addi %[[CH]], %c8 : index
// CHECK-NEXT: %[[HALF1:[a-zA-Z0-9_]+]] = memref.subview %arg1[%[[OUT]], %[[CH8]], 0, 0] [1, 8, 1, 16]
// CHECK-NEXT: %[[W2:[a-zA-Z0-9_]+]] = buckyball.bank_mvin_2d %[[HALF1]] %[[W1]] %c1_i64 %[[STRIDE64]] %c8_i64 %c8_i64 %c8_i64 %c16_i64
// CHECK-NEXT: %[[FIRST:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[CH]], %c0 : index
// CHECK-NEXT: %[[LAST:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[CH]], %c16 : index
// CHECK-NEXT: %[[SUM:[a-zA-Z0-9_]+]] = buckyball.bank_smatmul %[[PATCH]] %[[W2]] %[[CR]] %{{[a-zA-Z0-9_]+}} %[[FIRST]] %[[LAST]]
// CHECK-NEXT: scf.yield %[[PATCH]], %[[W2]], %[[SUM]] : i64, i64, i64
// CHECK-NEXT: }
// CHECK-NEXT: buckyball.bank_release %[[SRC]] : i64
// CHECK-NEXT: scf.yield %[[CHLOOP]]#0, %[[CHLOOP]]#1, %[[CHLOOP]]#2 : i64, i64, i64
// CHECK-NEXT: }
// CHECK: buckyball.bank_quant_i32_to_i8 %[[PANELS]]#2

// LLVM-LABEL: define void @grouped_cin32_cout40(
// LLVM: icmp sge i64 %[[STRIDE:[0-9]+]], 16
// LLVM-NEXT: icmp sle i64 %[[STRIDE]], 1016
// LLVM-NEXT: srem i64 %[[STRIDE]], 8
// LLVM: call void @abort()
// LLVM-LABEL: define void @_mlir_ciface_grouped_cin32_cout40(

func.func @grouped_cin32_cout40(%input: memref<1x1x1x32xi8>, %weight: memref<3x32x16x16xi8, strided<[?, ?, ?, ?], offset: ?>>, %bias: memref<40xi32>, %scale: memref<40xf32>, %lut: memref<1xi8>, %output: memref<1x1x1x40xi8>) {
  buckyball.mega_kernel %input %output : memref<1x1x1x32xi8> memref<1x1x1x40xi8> {
    buckyball.mega_conv2d %input %weight %bias %scale %lut %output {activation = 0 : i64, kernel = 1 : i64, outputScale = 1.0 : f32, padHigh = 0 : i64, padLow = 0 : i64, stride = 1 : i64} : memref<1x1x1x32xi8> memref<3x32x16x16xi8, strided<[?, ?, ?, ?], offset: ?>> memref<40xi32> memref<40xf32> memref<1xi8> memref<1x1x1x40xi8>
  }
  return
}
