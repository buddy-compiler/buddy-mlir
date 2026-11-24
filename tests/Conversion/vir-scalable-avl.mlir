// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true use-avl-loop=true vector-width=4' | FileCheck %s --check-prefix=UNKNOWN
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true use-avl-loop=true rvv-vlen=128 vector-width=4' | FileCheck %s --check-prefixes=CHECK,V128
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true use-avl-loop=true rvv-vlen=256 vector-width=4' | FileCheck %s --check-prefixes=CHECK,V256

func.func @add(%a: memref<?xf32>, %b: memref<?xf32>, %c: memref<?xf32>, %vl: index) {
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    %y = vir.load %b[] : memref<?xf32> -> !vir.vec<?xf32>
    %z = arith.addf %x, %y : !vir.vec<?xf32>
    vir.store %z, %c[] : !vir.vec<?xf32> -> memref<?xf32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @add
// CHECK: %[[S:.*]] = arith.muli {{.*}} : index
// CHECK: %[[REM:.*]] = arith.remsi
// CHECK: %[[END:.*]] = arith.subi {{.*}}, %[[REM]] : index
// CHECK: scf.for %[[IV:.*]] = {{.*}} to %[[END]] step %[[S]] {
// CHECK-NOT: llvm.intr.vp
// CHECK: vector.load {{.*}} vector<[4]xf32>
// CHECK: vector.load {{.*}} vector<[4]xf32>
// CHECK: vector.store
// CHECK: }
// CHECK: %[[TAIL:.*]] = arith.subi {{.*}}, %[[END]] : index
// CHECK: scf.if
// CHECK-NOT: scf.for
// CHECK: %[[EVL:.*]] = arith.index_cast %[[TAIL]] : index to i32
// CHECK: llvm.intr.vp.load{{.*}} %[[EVL]]
// CHECK: llvm.intr.vp.load{{.*}} %[[EVL]]
// CHECK: arith.addf {{.*}} : vector<[4]xf32>
// CHECK: llvm.intr.vp.store{{.*}} %[[EVL]]
// CHECK-NOT: scf.for
// CHECK: return

// Keep the accepted constant-length path ahead of AVL strip mining.
func.func @small(%a: memref<8xf32>, %b: memref<8xf32>) {
  %vl = arith.constant 8 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<8xf32> -> !vir.vec<?xf32>
    vir.store %x, %b[] : !vir.vec<?xf32> -> memref<8xf32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @small
// CHECK-NOT: scf.for
// CHECK-NOT: llvm.intr.vp
// CHECK: vector.load {{.*}} : memref<8xf32>, vector<8xf32>
// CHECK: vector.store {{.*}} : memref<8xf32>, vector<8xf32>
// CHECK-NOT: scf.for
// CHECK: return

// Ordered reductions must exclude inactive lanes rather than reducing poison.
func.func @sum(%a: memref<?xf32>, %o: memref<1xf32>, %vl: index) {
  %c0 = arith.constant 0 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    %s0 = memref.load %o[%c0] : memref<1xf32>
    %s = vir.reduce %x, %s0 {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %s, %o[%c0] : memref<1xf32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @sum
// CHECK: scf.for
// CHECK: vector.load
// CHECK: vector.reduction <add>
// CHECK: scf.if
// CHECK: llvm.intr.vp.load
// CHECK: llvm.intr.vp.reduce.fadd
// CHECK-NOT: scf.for
// CHECK: return

// Inactive integer divisors must not introduce undefined behavior.
func.func @div(%a: memref<?xi32>, %b: memref<?xi32>, %vl: index) {
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<?xi32> -> !vir.vec<?xi32>
    %y = vir.load %b[] : memref<?xi32> -> !vir.vec<?xi32>
    %z = arith.divsi %x, %y : !vir.vec<?xi32>
    vir.store %z, %a[] : !vir.vec<?xi32> -> memref<?xi32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @div
// CHECK: scf.for
// CHECK: arith.divsi
// CHECK: scf.if
// CHECK: llvm.intr.vp.sdiv
// CHECK-NOT: arith.divsi
// CHECK-NOT: scf.for
// CHECK: return

// Preserve the old path for non-unit-stride memory.
func.func @strided(%a: memref<?xf32, strided<[2], offset: ?>>, %b: memref<?xf32>, %vl: index) {
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<?xf32, strided<[2], offset: ?>> -> !vir.vec<?xf32>
    vir.store %x, %b[] : !vir.vec<?xf32> -> memref<?xf32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @strided
// CHECK-NOT: llvm.intr.vp
// CHECK: vector.transfer_read
// CHECK: scf.for
// CHECK: memref.load

// The active-length tail handles floating-point add and maxnum reductions
// only; a region with another kind keeps the fixed-VL lowering.
func.func @min_i32(%a: memref<?xi32>, %o: memref<1xi32>, %vl: index) {
  %c0 = arith.constant 0 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<?xi32> -> !vir.vec<?xi32>
    %s0 = memref.load %o[%c0] : memref<1xi32>
    %s = vir.reduce %x, %s0 {kind = "minsi"} : !vir.vec<?xi32>, i32 -> i32
    memref.store %s, %o[%c0] : memref<1xi32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @min_i32
// CHECK-NOT: llvm.intr.vp
// CHECK: return

func.func @prod(%a: memref<?xf32>, %o: memref<1xf32>, %vl: index) {
  %c0 = arith.constant 0 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    %s0 = memref.load %o[%c0] : memref<1xf32>
    %s = vir.reduce %x, %s0 {kind = "mul"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %s, %o[%c0] : memref<1xf32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @prod
// CHECK-NOT: llvm.intr.vp
// CHECK: return

// Preserve permission to reassociate, without relaxing ordered reductions.
func.func @reassoc_sum(%a: memref<?xf32>, %o: memref<1xf32>, %vl: index) {
  %c0 = arith.constant 0 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    %s0 = memref.load %o[%c0] : memref<1xf32>
    %s = vir.reduce %x, %s0 {kind = "add", reassoc} : !vir.vec<?xf32>, f32 -> f32
    memref.store %s, %o[%c0] : memref<1xf32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @reassoc_sum
// CHECK: llvm.call_intrinsic "llvm.vp.reduce.fadd"
// CHECK-SAME: fastmath<reassoc>
// CHECK-NOT: scf.for
// CHECK: return

// A compile-time length above one group but at most two uses a complete AVL
// loop. The bound changes with target VLEN, not a hard-coded lane count.
func.func @two_groups(%a: memref<24xf32>, %b: memref<24xf32>) {
  %vl = arith.constant 24 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<24xf32> -> !vir.vec<?xf32>
    vir.store %x, %b[] : !vir.vec<?xf32> -> memref<24xf32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @two_groups
// V128: arith.remsi
// V128: scf.for
// V128: vector.load
// V128: scf.if
// V128: llvm.intr.vp.load
// V256-NOT: arith.remsi
// V256: scf.for
// V256: arith.minui
// V256: llvm.intr.vp.load
// V256-NOT: scf.if
// CHECK: return

func.func @long_constant(%a: memref<1024xi32>, %b: memref<1024xi32>) {
  %vl = arith.constant 1024 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<1024xi32> -> !vir.vec<?xi32>
    %y = arith.divsi %x, %x : !vir.vec<?xi32>
    vir.store %y, %b[] : !vir.vec<?xi32> -> memref<1024xi32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @long_constant
// CHECK: arith.remsi
// CHECK: scf.for
// CHECK-NOT: llvm.intr.vp
// CHECK: arith.divsi
// CHECK: scf.if
// CHECK: llvm.intr.vp.sdiv
// CHECK-NOT: scf.for
// CHECK: return

// An unspecified target must not assume the two-group constant bound.
// UNKNOWN-LABEL: func.func @add
// UNKNOWN: arith.remsi
// UNKNOWN: scf.for
// UNKNOWN: vector.load
// UNKNOWN: scf.if
// UNKNOWN: llvm.intr.vp.load
// UNKNOWN-NOT: scf.for
// UNKNOWN: return
// UNKNOWN-LABEL: func.func @small
// UNKNOWN: scf.for
// UNKNOWN: vector.load
// UNKNOWN: scf.if
// UNKNOWN: llvm.intr.vp.load
// UNKNOWN: return

func.func @at_limit(%a: memref<32xf32>, %b: memref<32xf32>) {
  %vl = arith.constant 32 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<32xf32> -> !vir.vec<?xf32>
    vir.store %x, %b[] : !vir.vec<?xf32> -> memref<32xf32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @at_limit
// V128: arith.remsi
// V128: scf.for
// V128: vector.load
// V128: scf.if
// V128: llvm.intr.vp.load
// V256-NOT: arith.remsi
// V256: scf.for
// V256: arith.minui
// V256: llvm.intr.vp.load
// V256-NOT: scf.if
// CHECK: return

func.func @above_limit(%a: memref<33xf32>, %b: memref<33xf32>) {
  %vl = arith.constant 33 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<33xf32> -> !vir.vec<?xf32>
    vir.store %x, %b[] : !vir.vec<?xf32> -> memref<33xf32>
    vector.yield
  }
  return
}
// CHECK-LABEL: func.func @above_limit
// CHECK: arith.remsi
// CHECK: scf.for
// CHECK: vector.load
// CHECK: scf.if
// CHECK: llvm.intr.vp.load
// CHECK-NOT: scf.for
// CHECK: return
