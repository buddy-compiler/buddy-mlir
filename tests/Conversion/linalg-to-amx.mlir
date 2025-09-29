// RUN: buddy-opt %s -matmul-amx | FileCheck %s

// Test basic linalg.matmul to AMX conversion with static shapes
func.func @test_static_matmul_bf16(%A: memref<32x64xbf16>, %B: memref<64x48xbf16>, %C: memref<32x48xf32>) {
  // CHECK-LABEL: func.func @test_static_matmul_bf16
  // CHECK-NOT: linalg.matmul
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: amx.tile_zero
  // CHECK: amx.tile_store
  // CHECK: scf.for
  // CHECK: amx.tile_load
  // CHECK: amx.tile_load
  // CHECK: amx.tile_load
  // CHECK: amx.tile_mulf
  // CHECK: amx.tile_store
  
  linalg.matmul ins(%A, %B : memref<32x64xbf16>, memref<64x48xbf16>)
                outs(%C : memref<32x48xf32>)
  return
}

// Test with dynamic shapes - should still convert
func.func @test_dynamic_matmul_bf16(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>) {
  // CHECK-LABEL: func.func @test_dynamic_matmul_bf16
  // CHECK-NOT: linalg.matmul
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: amx.tile_zero
  // CHECK: amx.tile_store
  // CHECK: scf.for
  // CHECK: amx.tile_load
  // CHECK: amx.tile_load
  // CHECK: amx.tile_load
  // CHECK: amx.tile_mulf
  // CHECK: amx.tile_store
  
  linalg.matmul ins(%A, %B : memref<?x?xbf16>, memref<?x?xbf16>)
                outs(%C : memref<?x?xf32>)
  return
}

