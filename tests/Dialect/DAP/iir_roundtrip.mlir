// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

func.func @buddy_iir_f32(%in : memref<?xf32>, %filter : memref<?x?xf32>, %out : memref<?xf32>) -> () {
  // CHECK: dap.iir {{.*}} : memref<?xf32>, memref<?x?xf32>, memref<?xf32>
  dap.iir %in, %filter, %out : memref<?xf32>, memref<?x?xf32>, memref<?xf32>
  return
} 

func.func @buddy_iir_f64(%in : memref<?xf64>, %filter : memref<?xf64>, %out : memref<?xf64>) -> () {
  // CHECK: dap.iir {{.*}} : memref<?xf64>, memref<?xf64>, memref<?xf64>
  dap.iir %in, %filter, %out : memref<?xf64>, memref<?xf64>, memref<?xf64>
  return
}

func.func @buddy_iir_i8(%in : memref<?xi8>, %filter : memref<?xi8>, %out : memref<?xi8>) -> () {
  // CHECK: dap.iir {{.*}} : memref<?xi8>, memref<?xi8>, memref<?xi8>
  dap.iir %in, %filter, %out : memref<?xi8>, memref<?xi8>, memref<?xi8>
  return
}

func.func @buddy_iir_i32(%in : memref<?xi32>, %filter : memref<?xi32>, %out : memref<?xi32>) -> () {
  // CHECK: dap.iir {{.*}} : memref<?xi32>, memref<?xi32>, memref<?xi32>
  dap.iir %in, %filter, %out : memref<?xi32>, memref<?xi32>, memref<?xi32>
  return
}

func.func @buddy_iir_i64(%in : memref<?xi64>, %filter : memref<?xi64>, %out : memref<?xi64>) -> () {
  // CHECK: dap.iir {{.*}} : memref<?xi64>, memref<?xi64>, memref<?xi64>
  dap.iir %in, %filter, %out : memref<?xi64>, memref<?xi64>, memref<?xi64>
  return
}
