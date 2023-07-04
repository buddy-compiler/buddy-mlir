// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

func.func @buddy_resize2d_NEAREST_NEIGHBOUR_INTERPOLATION_f32(%input : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %output : memref<?x?xf32>) -> () {
  // CHECK: dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION %input, %horizontal_scaling_factor, %vertical_scaling_factor, %output : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}
