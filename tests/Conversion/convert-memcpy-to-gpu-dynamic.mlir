// RUN: buddy-opt -pass-pipeline="builtin.module(func.func(convert-memcpy-to-gpu{process-args=1}))" %s | FileCheck %s

// A dynamic size (the token count of a prefill) is read from the argument for
// its device copy and from the result for its host copy; an allocation keeps
// its dynamic sizes on the device.
func.func @scale(%arg0: memref<1x?xf32>) -> memref<1x?xf32> {
  // CHECK: %[[N:.*]] = memref.dim %arg0, %c1
  // CHECK: %[[D_ARG:.*]] = gpu.alloc  (%[[N]]) : memref<1x?xf32>
  // CHECK-NEXT: gpu.memcpy  %[[D_ARG]], %arg0
  %c1 = arith.constant 1 : index
  %n = memref.dim %arg0, %c1 : memref<1x?xf32>
  // CHECK: %[[D_ALLOC:.*]] = gpu.alloc  (%{{.*}}) : memref<1x?xf32>
  %alloc = memref.alloc(%n) : memref<1x?xf32>
  // CHECK: %[[M:.*]] = memref.dim %[[D_ALLOC]], %{{.*}}
  // CHECK: %[[H_RES:.*]] = memref.alloc(%[[M]]) : memref<1x?xf32>
  // CHECK: gpu.memcpy  %[[H_RES]], %[[D_ALLOC]]
  return %alloc : memref<1x?xf32>
}
