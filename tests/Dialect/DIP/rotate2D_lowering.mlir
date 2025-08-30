// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

func.func @buddy_rotate2d_f32(%input : memref<?x?xf32>, %angle : f32, %output : memref<?x?xf32>) -> () {
  // CHECK: dip.rotate_2d {{.*}} : memref<?x?xf32>, f32, memref<?x?xf32>
  dip.rotate_2d %input, %angle, %output : memref<?x?xf32>, f32, memref<?x?xf32>
  return
}
