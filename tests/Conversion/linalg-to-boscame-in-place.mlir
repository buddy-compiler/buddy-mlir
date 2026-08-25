// RUN: buddy-opt %s --lower-linalg-to-boscame --lower-bosc-ame | FileCheck %s

// Buddy Frontend bufferization leaves the matmul destination in place and
// lets later elementwise operations consume it.  It also places dynamic
// strides on packed function arguments.
func.func @buddy_in_place(
    %a: memref<4x16xi8, strided<[?, ?], offset: ?>>,
    %b: memref<16x4xi8, strided<[?, ?], offset: ?>>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %result = memref.alloc() : memref<4x4xf32>
  linalg.fill ins(%zero : f32) outs(%result : memref<4x4xf32>)
  linalg.matmul {cast = #linalg.type_fn<cast_signed>}
      ins(%a, %b : memref<4x16xi8, strided<[?, ?], offset: ?>>,
                    memref<16x4xi8, strided<[?, ?], offset: ?>>)
      outs(%result : memref<4x4xf32>)
  %value = memref.load %result[%c0, %c0] : memref<4x4xf32>
  return %value : f32
}

// CHECK-LABEL: func.func @buddy_in_place
// CHECK-NOT: linalg.matmul
// CHECK: llvm.call @llvm.riscv.bosc.mqma.b.mm
// CHECK: memref.load
