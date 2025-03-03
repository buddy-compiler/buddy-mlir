// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s


func.func @buddy_rotate4d_nhwc_f32(%inputImage : memref<?x?x?x?xf32>, %angle : f32, %outputImage : memref<?x?x?x?xf32>) -> () {
  // CHECK: dip.rotate_4d NHWC {{.*}} : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  dip.rotate_4d NHWC %inputImage, %angle, %outputImage : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  return
}

func.func @buddy_rotate4d_nhwc_f64(%inputImage : memref<?x?x?x?xf64>, %angle : f32, %outputImage : memref<?x?x?x?xf64>) -> () {
  // CHECK: dip.rotate_4d NHWC {{.*}} : memref<?x?x?x?xf64>, f32, memref<?x?x?x?xf64>
  dip.rotate_4d NHWC %inputImage, %angle, %outputImage : memref<?x?x?x?xf64>, f32, memref<?x?x?x?xf64>
  return
}

func.func @buddy_rotate4d_nhwc_i8(%inputImage : memref<?x?x?x?xi8>, %angle : f32, %outputImage : memref<?x?x?x?xi8>) -> () {
  // CHECK: dip.rotate_4d NHWC {{.*}} : memref<?x?x?x?xi8>, f32, memref<?x?x?x?xi8>
  dip.rotate_4d NHWC %inputImage, %angle, %outputImage : memref<?x?x?x?xi8>, f32, memref<?x?x?x?xi8>
  return
}

func.func @buddy_rotate4d_nhwc_i32(%inputImage : memref<?x?x?x?xi32>, %angle : f32, %outputImage : memref<?x?x?x?xi32>) -> () {
  // CHECK: dip.rotate_4d NHWC {{.*}} : memref<?x?x?x?xi32>, f32, memref<?x?x?x?xi32>
  dip.rotate_4d NHWC %inputImage, %angle, %outputImage : memref<?x?x?x?xi32>, f32, memref<?x?x?x?xi32>
  return
}

func.func @buddy_rotate4d_nhwc_i64(%inputImage : memref<?x?x?x?xi64>, %angle : f32, %outputImage : memref<?x?x?x?xi64>) -> () {
  // CHECK: dip.rotate_4d NHWC {{.*}} : memref<?x?x?x?xi64>, f32, memref<?x?x?x?xi64>
  dip.rotate_4d NHWC %inputImage, %angle, %outputImage : memref<?x?x?x?xi64>, f32, memref<?x?x?x?xi64>
  return
}

func.func @buddy_rotate4d_nchw_f32(%inputImage : memref<?x?x?x?xf32>, %angle : f32, %outputImage : memref<?x?x?x?xf32>) -> () {
  // CHECK: dip.rotate_4d NCHW {{.*}} : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  dip.rotate_4d NCHW %inputImage, %angle, %outputImage : memref<?x?x?x?xf32>, f32, memref<?x?x?x?xf32>
  return
}

func.func @buddy_rotate4d_nchw_f64(%inputImage : memref<?x?x?x?xf64>, %angle : f32, %outputImage : memref<?x?x?x?xf64>) -> () {
  // CHECK: dip.rotate_4d NCHW {{.*}} : memref<?x?x?x?xf64>, f32, memref<?x?x?x?xf64>
  dip.rotate_4d NCHW %inputImage, %angle, %outputImage : memref<?x?x?x?xf64>, f32, memref<?x?x?x?xf64>
  return
}

func.func @buddy_rotate4d_nchw_i8(%inputImage : memref<?x?x?x?xi8>, %angle : f32, %outputImage : memref<?x?x?x?xi8>) -> () {
  // CHECK: dip.rotate_4d NCHW {{.*}} : memref<?x?x?x?xi8>, f32, memref<?x?x?x?xi8>
  dip.rotate_4d NCHW %inputImage, %angle, %outputImage : memref<?x?x?x?xi8>, f32, memref<?x?x?x?xi8>
  return
}

func.func @buddy_rotate4d_nchw_i32(%inputImage : memref<?x?x?x?xi32>, %angle : f32, %outputImage : memref<?x?x?x?xi32>) -> () {
  // CHECK: dip.rotate_4d NCHW {{.*}} : memref<?x?x?x?xi32>, f32, memref<?x?x?x?xi32>
  dip.rotate_4d NCHW %inputImage, %angle, %outputImage : memref<?x?x?x?xi32>, f32, memref<?x?x?x?xi32>
  return
}

func.func @buddy_rotate4d_nchw_i64(%inputImage : memref<?x?x?x?xi64>, %angle : f32, %outputImage : memref<?x?x?x?xi64>) -> () {
  // CHECK: dip.rotate_4d NCHW {{.*}} : memref<?x?x?x?xi64>, f32, memref<?x?x?x?xi64>
  dip.rotate_4d NCHW %inputImage, %angle, %outputImage : memref<?x?x?x?xi64>, f32, memref<?x?x?x?xi64>
  return
}
