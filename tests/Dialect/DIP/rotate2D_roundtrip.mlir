// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

func.func @buddy_rotate2d_f32(%input : memref<?x?xf32>, %angle : f32, %output : memref<?x?xf32>) -> () {
  // CHECK: dip.rotate_2d {{.*}} : memref<?x?xf32>, f32, memref<?x?xf32>
  dip.rotate_2d %input, %angle, %output : memref<?x?xf32>, f32, memref<?x?xf32>
  return
}

func.func @buddy_rotate2d_f64(%input : memref<?x?xf64>, %angle : f32, %output : memref<?x?xf64>) -> () {
  // CHECK: dip.rotate_2d {{.*}} : memref<?x?xf64>, f32, memref<?x?xf64>
  dip.rotate_2d %input, %angle, %output : memref<?x?xf64>, f32, memref<?x?xf64>
  return
}

func.func @buddy_rotate2d_i8(%input : memref<?x?xi8>, %angle : f32, %output : memref<?x?xi8>) -> () {
  // CHECK: dip.rotate_2d {{.*}} : memref<?x?xi8>, f32, memref<?x?xi8>
  dip.rotate_2d %input, %angle, %output : memref<?x?xi8>, f32, memref<?x?xi8>
  return
}

func.func @buddy_rotate2d_i32(%input : memref<?x?xi32>, %angle : f32, %output : memref<?x?xi32>) -> () {
  // CHECK: dip.rotate_2d {{.*}} : memref<?x?xi32>, f32, memref<?x?xi32>
  dip.rotate_2d %input, %angle, %output : memref<?x?xi32>, f32, memref<?x?xi32>
  return
}

func.func @buddy_rotate2d_i64(%input : memref<?x?xi64>, %angle : f32, %output : memref<?x?xi64>) -> () {
  // CHECK: dip.rotate_2d {{.*}} : memref<?x?xi64>, f32, memref<?x?xi64>
  dip.rotate_2d %input, %angle, %output : memref<?x?xi64>, f32, memref<?x?xi64>
  return
}
