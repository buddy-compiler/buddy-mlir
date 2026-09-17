// RUN: buddy-opt %s --lower-linalg-to-boscame='target=qwen3-fpga' --lower-bosc-ame | FileCheck %s

// Buddy Frontend bufferization leaves the matmul destination in place and
// lets later elementwise operations consume it.  It also places dynamic
// leading strides on packed arguments, with a proven contiguous inner axis.
func.func @buddy_in_place(
    %a: memref<4x16xi8, strided<[?, 1], offset: ?>>,
    %b: memref<16x4xi8, strided<[?, 1], offset: ?>>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %result = memref.alloc() : memref<4x4xf32>
  linalg.fill ins(%zero : f32) outs(%result : memref<4x4xf32>)
  linalg.matmul {cast = #linalg.type_fn<cast_signed>}
      ins(%a, %b : memref<4x16xi8, strided<[?, 1], offset: ?>>,
                    memref<16x4xi8, strided<[?, 1], offset: ?>>)
      outs(%result : memref<4x4xf32>)
  %value = memref.load %result[%c0, %c0] : memref<4x4xf32>
  return %value : f32
}

// CHECK-LABEL: func.func @buddy_in_place
// CHECK-NOT: linalg.matmul
// The destination is reused in place: the accumulator is seeded from the
// original allocation and the result stays visible to the consumer.
// CHECK: bosc_ame.intr.fpga.mlce32.m
// CHECK: bosc_ame.intr.fpga.mqma.b.mm
// CHECK: bosc_ame.intr.fpga.msce32.m
// CHECK: memref.load
