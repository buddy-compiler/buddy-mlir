// RUN: buddy-opt %s -split-input-file -verify-diagnostics

// Invalid VIR ops: memory accesses and vir.set_vl regions.

// The lanes are along the last dim of the memref: it has one.
func.func @rank0(%a: memref<f32>) {
  // expected-error @below {{'vir.load' op operand #0 must be non-0-ranked.memref}}
  %x = vir.load %a[] : memref<f32> -> !vir.vec<?xf32>
  return
}

// -----

func.func @store_rank0(%x: !vir.vec<?xf32>, %a: memref<f32>) {
  // expected-error @below {{'vir.store' op operand #1 must be non-0-ranked.memref}}
  vir.store %x, %a[] : !vir.vec<?xf32> -> memref<f32>
  return
}

// -----

func.func @index_count(%a: memref<4x8xf32>, %i: index) {
  // expected-error @below {{needs no index or one per memref dim, 2}}
  %x = vir.load %a[%i] : memref<4x8xf32> -> !vir.vec<?xf32>
  return
}

// -----

func.func @store_index_count(%x: !vir.vec<?xf32>, %a: memref<4x8xf32>, %i: index) {
  // expected-error @below {{needs no index or one per memref dim, 2}}
  vir.store %x, %a[%i, %i, %i] : !vir.vec<?xf32> -> memref<4x8xf32>
  return
}

// -----

// The rows of a 2x? vector are along the dim before the last one.
func.func @rank_too_small(%a: memref<8xf32>) {
  // expected-error @below {{needs a memref of at least the rank of the dynamic vector, 2}}
  %x = vir.load %a[] : memref<8xf32> -> !vir.vec<2x?xf32>
  return
}

// -----

// The lowering maps the values of the region: its entry block has no
// arguments.
func.func @set_vl_block_arg(%o: memref<2x8xf32>) {
  %vl = arith.constant 8 : index
  %z = arith.constant 0.0 : f32
  // expected-error @below {{'vir.set_vl' op region must not have block arguments}}
  "vir.set_vl"(%vl) ({
  ^bb0(%x: !vir.vec<2x?xf32>):
    %dummy = vir.broadcast %z : f32 -> !vir.vec<?xf32>
    vir.store %x, %o[] : !vir.vec<2x?xf32> -> memref<2x8xf32>
    vector.yield
  }) : (index) -> ()
  return
}
