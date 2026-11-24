// RUN: buddy-opt %s -split-input-file -vir-schedule='verbose=true' \
// RUN: | FileCheck %s

// -vir-schedule orders the ops of a vir.set_vl region for fewer dynamic
// vectors live at once (MaxLive): an operand is computed right before its
// first user, the operand of most need first (Sethi-Ullman). The ops are
// only moved: the results are unchanged.

// (a + b) * (c + d), its loads first: a, b, c, d and a + b are live at the
// first add (5). Each sum computed from its loads: 4, at c + d.
func.func @loads_first(%m: memref<4x256xf32>, %o: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %vl = arith.constant 256 : index
  vir.set_vl %vl : index {
    %a = vir.load %m[%c0, %c0] : memref<4x256xf32> -> !vir.vec<?xf32>
    %b = vir.load %m[%c1, %c0] : memref<4x256xf32> -> !vir.vec<?xf32>
    %c = vir.load %m[%c2, %c0] : memref<4x256xf32> -> !vir.vec<?xf32>
    %d = vir.load %m[%c3, %c0] : memref<4x256xf32> -> !vir.vec<?xf32>
    %ab = arith.addf %a, %b : !vir.vec<?xf32>
    %cd = arith.addf %c, %d : !vir.vec<?xf32>
    %p = arith.mulf %ab, %cd : !vir.vec<?xf32>
    vir.store %p, %o[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vector.yield
  }
  return
}

// CHECK: vir-schedule: MaxLive 5 -> 4 in the block of vir.set_vl
// CHECK-LABEL: func.func @loads_first
// CHECK: vir.set_vl
// CHECK-NEXT: %[[A:.*]] = vir.load %{{.*}}[%c0, %c0]
// CHECK-NEXT: %[[B:.*]] = vir.load %{{.*}}[%c1, %c0]
// CHECK-NEXT: %[[AB:.*]] = arith.addf %[[A]], %[[B]]
// CHECK-NEXT: %[[C:.*]] = vir.load %{{.*}}[%c2, %c0]
// CHECK-NEXT: %[[D:.*]] = vir.load %{{.*}}[%c3, %c0]
// CHECK-NEXT: %[[CD:.*]] = arith.addf %[[C]], %[[D]]
// CHECK-NEXT: %[[P:.*]] = arith.mulf %[[AB]], %[[CD]]
// CHECK-NEXT: vir.store %[[P]]
// CHECK-NEXT: vector.yield

// -----

// A loop is one node, of the need of its body. x, loaded before the loop
// and used after it, is live across the loop (4: x and the body's 3);
// loaded after the loop, 3.
func.func @loop_node(%x: memref<256xf32>, %k: memref<64x256xf32>,
                     %o: memref<256xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %vl = arith.constant 256 : index
  vir.set_vl %vl : index {
    %xv = vir.load %x[%c0] : memref<256xf32> -> !vir.vec<?xf32>
    %init = vir.constant {value = 0.0 : f32} : !vir.vec<?xf32>
    %s = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init)
        -> (!vir.vec<?xf32>) {
      %kv = vir.load %k[%i, %c0] : memref<64x256xf32> -> !vir.vec<?xf32>
      %next = arith.addf %acc, %kv : !vir.vec<?xf32>
      scf.yield %next : !vir.vec<?xf32>
    }
    %r = arith.addf %s, %xv : !vir.vec<?xf32>
    vir.store %r, %o[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vector.yield
  }
  return
}

// CHECK: vir-schedule: MaxLive 4 -> 3 in the block of vir.set_vl
// CHECK-LABEL: func.func @loop_node
// CHECK: vir.set_vl
// CHECK-NEXT: vir.constant
// CHECK-NEXT: scf.for
// CHECK: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: vir.load %{{.*}}[%c0] : memref<256xf32>
// CHECK-NEXT: arith.addf
// CHECK-NEXT: vir.store

// -----

// A read stays after the write before it, and a write after the reads
// before it: y is not loaded after the store to %t, which may be %b, nor z
// before it. Unchanged.
func.func @memory_order(%a: memref<256xf32>, %b: memref<256xf32>,
                        %t: memref<256xf32>, %o: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 256 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[%c0] : memref<256xf32> -> !vir.vec<?xf32>
    %y = vir.load %b[%c0] : memref<256xf32> -> !vir.vec<?xf32>
    vir.store %x, %t[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    %z = vir.load %t[%c0] : memref<256xf32> -> !vir.vec<?xf32>
    %w = arith.addf %y, %z : !vir.vec<?xf32>
    vir.store %w, %o[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vector.yield
  }
  return
}

// CHECK-NOT: vir-schedule:
// CHECK-LABEL: func.func @memory_order
// CHECK: vir.set_vl
// CHECK-NEXT: %[[X:.*]] = vir.load %arg0
// CHECK-NEXT: %[[Y:.*]] = vir.load %arg1
// CHECK-NEXT: vir.store %[[X]], %arg2
// CHECK-NEXT: %[[Z:.*]] = vir.load %arg2
// CHECK-NEXT: arith.addf %[[Y]], %[[Z]]

// -----

// Another order of the same MaxLive (the load before the broadcast): the
// original order is kept.
func.func @no_gain(%a: memref<256xf32>, %o: memref<256xf32>, %f: f32) {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 256 : index
  vir.set_vl %vl : index {
    %s = vir.broadcast %f : f32 -> !vir.vec<?xf32>
    %x = vir.load %a[%c0] : memref<256xf32> -> !vir.vec<?xf32>
    %y = arith.mulf %x, %s : !vir.vec<?xf32>
    vir.store %y, %o[%c0] : !vir.vec<?xf32> -> memref<256xf32>
    vector.yield
  }
  return
}

// CHECK: vir-schedule: MaxLive 3 -> 3, original order kept in the block of vir.set_vl
// CHECK-LABEL: func.func @no_gain
// CHECK: vir.set_vl
// CHECK-NEXT: vir.broadcast
// CHECK-NEXT: vir.load

// -----

// An op that is not speculatable (a division of a divisor not known
// nonzero) stays after the last op before it that may not return, the call
// that checks the divisor: with a divisor 0, the assertion stops the
// program before the division. The rest is ordered: 4 -> 3.
func.func private @check_divisor(%d: i32) -> i32

func.func @guarded_division(%x: i32, %d: i32, %s1: i32, %s2: i32,
                            %out: memref<4xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 4 : index
  %result = vir.set_vl %vl : index {
    %g = func.call @check_divisor(%d) : (i32) -> i32
    %q = arith.divsi %x, %d : i32
    %a = vir.broadcast %q : i32 -> !vir.vec<?xi32>
    %b = vir.broadcast %s1 : i32 -> !vir.vec<?xi32>
    %c = vir.broadcast %s2 : i32 -> !vir.vec<?xi32>
    %ab = arith.addi %a, %b : !vir.vec<?xi32>
    %sum = arith.addi %ab, %c : !vir.vec<?xi32>
    vir.store %sum, %out[%c0] : !vir.vec<?xi32> -> memref<4xi32>
    vector.yield %g : i32
  } -> i32
  return %result : i32
}

// CHECK: vir-schedule: MaxLive 4 -> 3 in the block of
// CHECK-LABEL: func.func @guarded_division
// CHECK: vir.set_vl
// CHECK-NEXT: func.call @check_divisor
// CHECK-NEXT: arith.divsi
// CHECK-NEXT: vir.broadcast
// CHECK-NEXT: vir.broadcast
// CHECK-NEXT: arith.addi
// CHECK-NEXT: vir.broadcast
// CHECK-NEXT: arith.addi
// CHECK-NEXT: vir.store

// -----

// The same after a loop that may not end (an scf.while, which loops while
// %d is 0): the division by %d stays after it.
func.func @division_after_while(%x: i32, %d: i32, %s1: i32, %s2: i32,
                                %out: memref<4xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %vl = arith.constant 4 : index
  %result = vir.set_vl %vl : index {
    %w = scf.while (%i = %d) : (i32) -> i32 {
      %zero = arith.constant 0 : i32
      %more = arith.cmpi eq, %i, %zero : i32
      scf.condition(%more) %i : i32
    } do {
    ^bb0(%i: i32):
      scf.yield %i : i32
    }
    %q = arith.divsi %x, %d : i32
    %a = vir.broadcast %q : i32 -> !vir.vec<?xi32>
    %b = vir.broadcast %s1 : i32 -> !vir.vec<?xi32>
    %c = vir.broadcast %s2 : i32 -> !vir.vec<?xi32>
    %ab = arith.addi %a, %b : !vir.vec<?xi32>
    %sum = arith.addi %ab, %c : !vir.vec<?xi32>
    vir.store %sum, %out[%c0] : !vir.vec<?xi32> -> memref<4xi32>
    vector.yield %w : i32
  } -> i32
  return %result : i32
}

// CHECK: vir-schedule: MaxLive 4 -> 3 in the block of
// CHECK-LABEL: func.func @division_after_while
// CHECK: vir.set_vl
// CHECK-NEXT: scf.while
// CHECK: arith.divsi
// CHECK-NEXT: vir.broadcast
