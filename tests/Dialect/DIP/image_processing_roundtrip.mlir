// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

func.func @buddy_corr2d_CONSTANT_PADDING(%input : memref<?x?xf32>, %identity : memref<?x?xf32>, %output : memref<?x?xf32>, %kernelAnchorX : index, %kernelAnchorY : index, %c : f32) -> () {
  // CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  dip.corr_2d <CONSTANT_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @buddy_corr2d_REPLICATE_PADDING(%input : memref<?x?xf32>, %identity : memref<?x?xf32>, %output : memref<?x?xf32>, %kernelAnchorX : index, %kernelAnchorY : index, %c : f32) -> () {
  // CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  dip.corr_2d <REPLICATE_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @buddy_resize2d_NEAREST_NEIGHBOUR_INTERPOLATION(%input : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %output : memref<?x?xf32>) -> () {
  // CHECK: dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION %input, %horizontal_scaling_factor, %vertical_scaling_factor, %output : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}

func.func @buddy_resize2d_BILINEAR_INTERPOLATION(%input : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %output : memref<?x?xf32>) -> () {
  // CHECK: dip.resize_2d BILINEAR_INTERPOLATION{{.*}} : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  dip.resize_2d BILINEAR_INTERPOLATION %input, %horizontal_scaling_factor, %vertical_scaling_factor, %output : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}

func.func @buddy_rotate2d(%input : memref<?x?xf32>, %angle : f32, %output : memref<?x?xf32>) -> () {
  // CHECK: dip.rotate_2d {{.*}} : memref<?x?xf32>, f32, memref<?x?xf32>
  dip.rotate_2d %input, %angle, %output : memref<?x?xf32>, f32, memref<?x?xf32>
  return
}
