// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s


func.func @buddy_rotate4d_nhwc_f32(%inputImage : memref<?x?x?x?xf32>, %angle : f32, %outputImage : memref<?x?x?x?xf32>) -> () {
  // CHECK: dip.rotate_4d NHWC {{.*}} : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  dip.rotate_4d NHWC %inputImage, %angle, %outputImage : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  return
}

func.func @buddy_rotate4d_nchw(%inputImage : memref<?x?x?x?xf32>, %angle : f32, %outputImage : memref<?x?x?x?xf32>) -> () {
  // CHECK: dip.rotate_4d NCHW {{.*}} : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  dip.rotate_4d NCHW %inputImage, %angle, %outputImage : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  return
}
