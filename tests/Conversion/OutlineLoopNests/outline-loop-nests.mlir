// RUN: buddy-opt %s -outline-loop-nests | FileCheck %s

// Every top-level loop nest moves into a function of its own; the values it
// reads become arguments and its results the results of the call.
func.func @forward(%a: memref<8xf32>, %b: memref<8xf32>, %scale: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.parallel (%i) = (%c0) to (%c8) step (%c1) {
    %x = memref.load %a[%i] : memref<8xf32>
    %y = arith.mulf %x, %scale : f32
    memref.store %y, %b[%i] : memref<8xf32>
    scf.reduce
  }
  %zero = arith.constant 0.0 : f32
  %sum = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> f32 {
    %x = memref.load %b[%i] : memref<8xf32>
    %next = arith.addf %acc, %x : f32
    scf.yield %next : f32
  }
  // Captures also used outside the nests must keep their original value.
  %result = arith.addf %sum, %scale : f32
  return %result : f32
}

// CHECK-LABEL: func.func private @forward_nest0(
// CHECK-SAME: llvm.passthrough = ["noinline"]
// CHECK: scf.parallel
// CHECK-LABEL: func.func private @forward_nest1(
// CHECK-SAME: -> f32
// CHECK-SAME: llvm.passthrough = ["noinline"]
// CHECK: scf.for
// CHECK-LABEL: func.func @forward(
// CHECK-NOT: llvm.passthrough
// CHECK-NOT: scf.parallel
// CHECK-NOT: scf.for
// CHECK: call @forward_nest0(
// CHECK: %[[SUM:.*]] = call @forward_nest1(
// CHECK: %[[RESULT:.*]] = arith.addf %[[SUM]], %arg2 : f32
// CHECK: return %[[RESULT]] : f32
