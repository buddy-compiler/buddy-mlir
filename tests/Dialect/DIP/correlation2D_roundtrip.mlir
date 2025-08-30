// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

func.func @buddy_corr2d_CONSTANT_PADDING_f32(%input : memref<?x?xf32>, %identity : memref<?x?xf32>, %output : memref<?x?xf32>, %kernelAnchorX : index, %kernelAnchorY : index, %c : f32) -> () {
  // CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  dip.corr_2d <CONSTANT_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @buddy_corr2d_CONSTANT_PADDING_f64(%input : memref<?x?xf64>, %identity : memref<?x?xf64>, %output : memref<?x?xf64>, %kernelAnchorX : index, %kernelAnchorY : index, %c : f64) -> () {
  // CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, index, index, f64
  dip.corr_2d <CONSTANT_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, index, index, f64
  return
}

func.func @buddy_corr2d_CONSTANT_PADDING_i8(%input : memref<?x?xi8>, %identity : memref<?x?xi8>, %output : memref<?x?xi8>, %kernelAnchorX : index, %kernelAnchorY : index, %c : i8) -> () {
  // CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi8>, index, index, i8
  dip.corr_2d <CONSTANT_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi8>, index, index, i8
  return
}

func.func @buddy_corr2d_CONSTANT_PADDING_i32(%input : memref<?x?xi32>, %identity : memref<?x?xi32>, %output : memref<?x?xi32>, %kernelAnchorX : index, %kernelAnchorY : index, %c : i32) -> () {
  // CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, index, index, i32
  dip.corr_2d <CONSTANT_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, index, index, i32
  return
}

func.func @buddy_corr2d_CONSTANT_PADDING_i64(%input : memref<?x?xi64>, %identity : memref<?x?xi64>, %output : memref<?x?xi64>, %kernelAnchorX : index, %kernelAnchorY : index, %c : i64) -> () {
  // CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xi64>, memref<?x?xi64>, memref<?x?xi64>, index, index, i64
  dip.corr_2d <CONSTANT_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xi64>, memref<?x?xi64>, memref<?x?xi64>, index, index, i64
  return
}

func.func @buddy_corr2d_REPLICATE_PADDING_f32(%input : memref<?x?xf32>, %identity : memref<?x?xf32>, %output : memref<?x?xf32>, %kernelAnchorX : index, %kernelAnchorY : index, %c : f32) -> () {
  // CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  dip.corr_2d <REPLICATE_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @buddy_corr2d_REPLICATE_PADDING_f64(%input : memref<?x?xf64>, %identity : memref<?x?xf64>, %output : memref<?x?xf64>, %kernelAnchorX : index, %kernelAnchorY : index, %c : f64) -> () {
  // CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, index, index, f64
  dip.corr_2d <REPLICATE_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, index, index, f64
  return
}

func.func @buddy_corr2d_REPLICATE_PADDING_i8(%input : memref<?x?xi8>, %identity : memref<?x?xi8>, %output : memref<?x?xi8>, %kernelAnchorX : index, %kernelAnchorY : index, %c : i8) -> () {
  // CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi8>, index, index, i8
  dip.corr_2d <REPLICATE_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi8>, index, index, i8
  return
}

func.func @buddy_corr2d_REPLICATE_PADDING_i32(%input : memref<?x?xi32>, %identity : memref<?x?xi32>, %output : memref<?x?xi32>, %kernelAnchorX : index, %kernelAnchorY : index, %c : i32) -> () {
  // CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, index, index, i32
  dip.corr_2d <REPLICATE_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, index, index, i32
  return
}

func.func @buddy_corr2d_REPLICATE_PADDING_i64(%input : memref<?x?xi64>, %identity : memref<?x?xi64>, %output : memref<?x?xi64>, %kernelAnchorX : index, %kernelAnchorY : index, %c : i64) -> () {
  // CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xi64>, memref<?x?xi64>, memref<?x?xi64>, index, index, i64
  dip.corr_2d <REPLICATE_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<?x?xi64>, memref<?x?xi64>, memref<?x?xi64>, index, index, i64
  return
}
