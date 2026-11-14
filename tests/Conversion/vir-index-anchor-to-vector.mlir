// RUN: buddy-opt %s -lower-vir-to-vector='rvv-vlen=1024 verbose=true' \
// RUN: | FileCheck %s --check-prefix=FIXED
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256 verbose=true' \
// RUN: | FileCheck %s --check-prefix=SCALABLE

// The automatic vector size of a region of index vectors takes the index
// width of the LLVM lowering (64 bits) as the element width.

// One vector live: LMUL 8, 8 * 1024 / 64 = 128 lanes; vector<[8]xindex>
// (LMUL 8 * 64 / 64) when scalable.
func.func @offsets(%a: memref<?xindex>, %n: index, %d: index) {
  vir.set_vl %n : index {
    %x = vir.load %a[] : memref<?xindex> -> !vir.vec<?xindex>
    %b = vir.broadcast %d : index -> !vir.vec<?xindex>
    %y = arith.addi %x, %b : !vir.vec<?xindex>
    vir.store %y, %a[] : !vir.vec<?xindex> -> memref<?xindex>
    vector.yield
  }
  return
}

// FIXED: Element Bit Width (Width_Ele): 64
// FIXED: Calculated LMUL = 8, Size_vec = (8 * 1024) / 64 = 128
// FIXED-LABEL: func.func @offsets
// FIXED: vector.load {{.*}} : memref<?xindex>, vector<128xindex>
// SCALABLE: Element Bit Width (Width_Ele): 64
// SCALABLE: Scalable: N = max(1, (32 * 64) / 256) = 8, vector<[8]xindex>
// SCALABLE-LABEL: func.func @offsets
// SCALABLE: vector.load {{.*}} : memref<?xindex>, vector<[8]xindex>

// A set_vl with results: the rows of index vectors shuffled.
func.func @rows(%a: memref<2x2xindex>, %o: memref<2x2xindex>) -> i32 {
  %vl = arith.constant 2 : index
  %zero = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x2xindex> -> !vir.vec<2x?xindex>
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xindex>
    vir.store %y, %o[] : !vir.vec<2x?xindex> -> memref<2x2xindex>
    vector.yield %zero : i32
  } -> i32
  return %r : i32
}

// FIXED-LABEL: func.func @rows
// FIXED: vector.shuffle %{{.*}}, %{{.*}} [2, 3, 0, 1] : vector<4xindex>, vector<4xindex>
// SCALABLE-LABEL: func.func @rows
// SCALABLE: llvm.call_intrinsic "llvm.vector.splice.left"({{.*}}) : (vector<[1]xi64>, vector<[1]xi64>, i32) -> vector<[1]xi64>
