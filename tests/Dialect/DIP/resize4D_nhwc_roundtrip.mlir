// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

func.func @buddy_resize4d_nhwc_NEAREST_NEIGHBOUR_INTERPOLATION_f32(%input : memref<?x?x?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %output : memref<?x?x?x?xf32>) -> () {
  // CHECK: dip.resize_4d_nhwc NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  dip.resize_4d_nhwc NEAREST_NEIGHBOUR_INTERPOLATION %input, %horizontal_scaling_factor, %vertical_scaling_factor, %output : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  return
}

func.func @buddy_resize4d_nhwc_NEAREST_NEIGHBOUR_INTERPOLATION_f64(%input : memref<?x?x?x?xf64>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %output : memref<?x?x?x?xf64>) -> () {
  // CHECK: dip.resize_4d_nhwc NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?x?x?xf64>, f32, f32, memref<?x?x?x?xf64>
  dip.resize_4d_nhwc NEAREST_NEIGHBOUR_INTERPOLATION %input, %horizontal_scaling_factor, %vertical_scaling_factor, %output : memref<?x?x?x?xf64>, f32, f32, memref<?x?x?x?xf64>
  return
}

func.func @buddy_resize4d_nhwc_BILINEAR_INTERPOLATION_f32(%input : memref<?x?x?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %output : memref<?x?x?x?xf32>) -> () {
  // CHECK: dip.resize_4d_nhwc BILINEAR_INTERPOLATION{{.*}} : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  dip.resize_4d_nhwc BILINEAR_INTERPOLATION %input, %horizontal_scaling_factor, %vertical_scaling_factor, %output : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  return
}

func.func @buddy_resize4d_nhwc_BILINEAR_INTERPOLATION_f64(%input : memref<?x?x?x?xf64>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %output : memref<?x?x?x?xf64>) -> () {
  // CHECK: dip.resize_4d_nhwc BILINEAR_INTERPOLATION{{.*}} : memref<?x?x?x?xf64>, f32, f32, memref<?x?x?x?xf64>
  dip.resize_4d_nhwc BILINEAR_INTERPOLATION %input, %horizontal_scaling_factor, %vertical_scaling_factor, %output : memref<?x?x?x?xf64>, f32, f32, memref<?x?x?x?xf64>
  return
}
