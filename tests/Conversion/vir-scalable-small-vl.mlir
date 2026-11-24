// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=128 vector-width=4' | FileCheck %s --check-prefixes=SMALL,V128
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256 vector-width=4' | FileCheck %s --check-prefixes=SMALL,V256
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true vector-width=4' | FileCheck %s --check-prefix=UNKNOWN
// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4 rvv-vlen=256' | FileCheck %s --check-prefix=FIXED

// At minimum VLEN 128, vector<[4]xf32> can hold exactly eight lanes.
func.func @exact(%a: memref<8xf32>, %b: memref<8xf32>) {
  %vl = arith.constant 8 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<8xf32> -> !vir.vec<?xf32>
    vir.store %x, %b[] : !vir.vec<?xf32> -> memref<8xf32>
    vector.yield
  }
  return
}
// SMALL-LABEL: func.func @exact
// SMALL-NOT: scf.for
// SMALL-NOT: affine.for
// SMALL: vector.load {{.*}} : memref<8xf32>, vector<8xf32>
// SMALL: vector.store {{.*}} : memref<8xf32>, vector<8xf32>
// SMALL-NOT: scf.for
// SMALL-NOT: affine.for
// SMALL: return
// UNKNOWN-LABEL: func.func @exact
// UNKNOWN: scf.for
// UNKNOWN: vector.load {{.*}} vector<[4]xf32>
// FIXED-LABEL: func.func @exact
// FIXED: affine.for
// FIXED: vector.load {{.*}} vector<4xf32>

// A non-power-of-two AVL uses the exact length, without touching extra lanes.
func.func @odd_rows(%a: memref<2x3xf32>, %b: memref<2x3xf32>) {
  %vl = arith.constant 3 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x3xf32> -> !vir.vec<2x?xf32>
    vir.store %x, %b[] : !vir.vec<2x?xf32> -> memref<2x3xf32>
    vector.yield
  }
  return
}
// SMALL-LABEL: func.func @odd_rows
// SMALL: affine.for
// SMALL-NOT: scf.for
// SMALL: vector.load {{.*}} vector<3xf32>
// SMALL: vector.store {{.*}} vector<3xf32>
// SMALL-NOT: scf.for
// SMALL: return

func.func @above_128(%a: memref<9xf32>, %b: memref<9xf32>) {
  %vl = arith.constant 9 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<9xf32> -> !vir.vec<?xf32>
    vir.store %x, %b[] : !vir.vec<?xf32> -> memref<9xf32>
    vector.yield
  }
  return
}
// V128-LABEL: func.func @above_128
// V128: scf.for
// V128: vector.load {{.*}} vector<[4]xf32>
// V256-LABEL: func.func @above_128
// V256-NOT: scf.for
// V256: vector.load {{.*}} vector<9xf32>
// V256-NOT: scf.for
// V256: return

// Unknown and empty lengths must stay on the existing path.
func.func @dynamic(%a: memref<?xf32>, %b: memref<?xf32>, %vl: index) {
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    vir.store %x, %b[] : !vir.vec<?xf32> -> memref<?xf32>
    vector.yield
  }
  return
}
// SMALL-LABEL: func.func @dynamic
// SMALL: scf.for
// SMALL: vector.load {{.*}} vector<[4]xf32>
// SMALL: scf.for

func.func @empty(%a: memref<0xf32>, %b: memref<0xf32>) {
  %vl = arith.constant 0 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<0xf32> -> !vir.vec<?xf32>
    vir.store %x, %b[] : !vir.vec<?xf32> -> memref<0xf32>
    vector.yield
  }
  return
}
// SMALL-LABEL: func.func @empty
// SMALL: scf.for
// SMALL: vector.load {{.*}} vector<[4]xf32>

// Preserve the ordered reduction contract when forming the fixed vector.
func.func @ordered_sum(%a: memref<3xf32>, %o: memref<1xf32>) {
  %vl = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<3xf32> -> !vir.vec<?xf32>
    %s0 = memref.load %o[%c0] : memref<1xf32>
    %s = vir.reduce %x, %s0 {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %s, %o[%c0] : memref<1xf32>
    vector.yield
  }
  return
}
// SMALL-LABEL: func.func @ordered_sum
// SMALL-NOT: scf.for
// SMALL: vector.reduction <add>, {{.*}} : vector<3xf32> into f32
// SMALL-NOT: scf.for
// SMALL: return
