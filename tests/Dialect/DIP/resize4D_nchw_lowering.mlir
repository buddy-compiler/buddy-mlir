// RUN: buddy-opt --lower-dip %s | buddy-opt | FileCheck %s

func.func @buddy_resize4d_nchw_NEAREST_NEIGHBOUR_INTERPOLATION_f32(%input : memref<?x?x?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %output : memref<?x?x?x?xf32>) -> () {
  // CHECK: memref.store %57, %arg3[%arg4, %arg5, %56, %54] : memref<?x?x?x?xf32>
  dip.resize_4d_nchw NEAREST_NEIGHBOUR_INTERPOLATION %input, %horizontal_scaling_factor, %vertical_scaling_factor, %output : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  return
}
