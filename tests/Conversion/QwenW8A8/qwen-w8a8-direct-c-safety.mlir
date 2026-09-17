// RUN: buddy-opt %s --lower-linalg-to-boscame=target=qwen3-fpga | FileCheck %s

func.func @alias_direct_c(%raw: memref<512xi8>, %B: memref<32x4xi8>) {
 %i = arith.constant 0 : index
 %A = memref.view %raw[%i][] : memref<512xi8> to memref<4x32xi8>
 %out = memref.view %raw[%i][] : memref<512xi8> to memref<4x4xf32>
 %C = memref.alloc() : memref<4x4xf32>
 %zero = arith.constant 0.0 : f32
 linalg.fill ins(%zero : f32) outs(%C : memref<4x4xf32>)
 linalg.matmul ins(%A, %B : memref<4x32xi8>, memref<32x4xi8>) outs(%C : memref<4x4xf32>)
 memref.copy %C, %out : memref<4x4xf32> to memref<4x4xf32>
 memref.dealloc %C : memref<4x4xf32>
 return
}

func.func @read_after_copy(%A: memref<4x32xi8>, %B: memref<32x4xi8>, %out: memref<4x4xf32>) -> f32 {
 // Prove no alias so only the temporary's extra consumer prevents elision.
 %a, %b, %dst = memref.distinct_objects %A, %B, %out : memref<4x32xi8>, memref<32x4xi8>, memref<4x4xf32>
 %i = arith.constant 0 : index
 %C = memref.alloc() : memref<4x4xf32>
 %zero = arith.constant 0.0 : f32
 linalg.fill ins(%zero : f32) outs(%C : memref<4x4xf32>)
 linalg.matmul ins(%a, %b : memref<4x32xi8>, memref<32x4xi8>) outs(%C : memref<4x4xf32>)
 memref.copy %C, %dst : memref<4x4xf32> to memref<4x4xf32>
 %x = memref.load %C[%i, %i] : memref<4x4xf32>
 memref.dealloc %C : memref<4x4xf32>
 return %x : f32
}

func.func @dynamic_stride(%A: memref<4x32xi8, strided<[64, ?]>>, %B: memref<32x4xi8>, %out: memref<4x4xf32>) {
 // Prove no alias so an unknown inner stride is the only reason for fallback.
 %a, %b, %dst = memref.distinct_objects %A, %B, %out : memref<4x32xi8, strided<[64, ?]>>, memref<32x4xi8>, memref<4x4xf32>
 %C = memref.alloc() : memref<4x4xf32>
 %zero = arith.constant 0.0 : f32
 linalg.fill ins(%zero : f32) outs(%C : memref<4x4xf32>)
 linalg.matmul ins(%a, %b : memref<4x32xi8, strided<[64, ?]>>, memref<32x4xi8>) outs(%C : memref<4x4xf32>)
 memref.copy %C, %dst : memref<4x4xf32> to memref<4x4xf32>
 memref.dealloc %C : memref<4x4xf32>
 return
}

// CHECK-LABEL: func.func @alias_direct_c
// CHECK: %[[C:.*]] = memref.alloc
// CHECK: linalg.fill {{.*}} outs(%[[C]]
// CHECK: linalg.matmul
// CHECK: memref.copy %[[C]]
// CHECK-LABEL: func.func @read_after_copy
// CHECK: %[[TMP:.*]] = memref.alloc
// CHECK: linalg.matmul
// CHECK: memref.copy %[[TMP]]
// CHECK: memref.load %[[TMP]]
// CHECK-LABEL: func.func @dynamic_stride
// CHECK: linalg.matmul
// CHECK-NOT: bosc_ame.mlae8.m
