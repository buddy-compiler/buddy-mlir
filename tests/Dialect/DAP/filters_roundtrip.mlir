// RUN: buddy-opt -verify-diagnostics %s | buddy-opt | FileCheck %s

func.func @buddy_iir(%in : memref<?xf32>, %filter : memref<?x?xf32>, %out : memref<?xf32>) -> () {
  // CHECK: dap.iir {{.*}} : memref<?xf32>, memref<?x?xf32>, memref<?xf32>
  dap.iir %in, %filter, %out : memref<?xf32>, memref<?x?xf32>, memref<?xf32>
  return
} 

func.func @buddy_fir(%in : memref<?xf32>, %filter : memref<?xf32>, %out : memref<?xf32>) -> () {
  // CHECK: dap.fir {{.*}} : memref<?xf32>, memref<?xf32>, memref<?xf32>
  dap.fir %in, %filter, %out : memref<?xf32>, memref<?xf32>, memref<?xf32>
  return
}
