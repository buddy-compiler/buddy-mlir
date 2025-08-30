// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

func.func @buddy_fir_f32(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  // CHECK: dap.fir {{.*}} : memref<?xf32>, memref<?xf32>, memref<?xf32>
  dap.fir %in, %filter, %out : memref<?xf32>, memref<?xf32>, memref<?xf32>
  return
}

func.func @buddy_fir_f64(%in : memref<?xf64>, %filter : memref<?xf64>, %out : memref<?xf64>) -> () {
  // CHECK: dap.fir {{.*}} : memref<?xf64>, memref<?xf64>, memref<?xf64>
  dap.fir %in, %filter, %out : memref<?xf64>, memref<?xf64>, memref<?xf64>
  return
}

func.func @buddy_fir_i8(%in : memref<?xi8>, %filter : memref<?xi8>, %out : memref<?xi8>) -> () {
  // CHECK: dap.fir {{.*}} : memref<?xi8>, memref<?xi8>, memref<?xi8>
  dap.fir %in, %filter, %out : memref<?xi8>, memref<?xi8>, memref<?xi8>
  return
}

func.func @buddy_fir_i32(%in : memref<?xi32>, %filter : memref<?xi32>, %out : memref<?xi32>) -> () {
  // CHECK: dap.fir {{.*}} : memref<?xi32>, memref<?xi32>, memref<?xi32>
  dap.fir %in, %filter, %out : memref<?xi32>, memref<?xi32>, memref<?xi32>
  return
}

func.func @buddy_fir_i64(%in : memref<?xi64>, %filter : memref<?xi64>, %out : memref<?xi64>) -> () {
  // CHECK: dap.fir {{.*}} : memref<?xi64>, memref<?xi64>, memref<?xi64>
  dap.fir %in, %filter, %out : memref<?xi64>, memref<?xi64>, memref<?xi64>
  return
}
