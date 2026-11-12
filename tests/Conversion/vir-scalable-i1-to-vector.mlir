// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN: | FileCheck %s
// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4 use-scalable=true rvv-vlen=256' \
// RUN: | FileCheck %s --check-prefix=WIDTH4

// A memref of i1 holds a byte per element, a vector of i1 a bit per lane: a
// scalable vector accesses the bytes of the memref (a vector<[N]xi8> at the
// address of the first element), truncated to / extended from i1. The
// scalar tail loop accesses the i1 elements.

// CHECK-LABEL: func.func @copy_bool
// CHECK: scf.for %[[I:.*]] = {{.*}} {
// CHECK:   %{{.*}}, %[[OFF:.*]], %{{.*}}, %[[STRIDE:.*]] = memref.extract_strided_metadata %[[A:.*]] : memref<128xi1>
// CHECK:   %[[SCALED:.*]] = arith.muli %[[I]], %[[STRIDE]] : index
// CHECK:   %[[LIN:.*]] = arith.addi %[[OFF]], %[[SCALED]] : index
// CHECK:   %[[P:.*]] = memref.extract_aligned_pointer_as_index %[[A]] : memref<128xi1> -> index
// CHECK:   %[[ADDR:.*]] = arith.addi %[[P]], %[[LIN]] : index
// CHECK:   %[[BITS:.*]] = arith.index_cast %[[ADDR]] : index to i64
// CHECK:   %[[PTR:.*]] = llvm.inttoptr %[[BITS]] : i64 to !llvm.ptr
// CHECK:   %[[BYTES:.*]] = llvm.load %[[PTR]] <alignment = 1> : !llvm.ptr -> vector<[32]xi8>
// CHECK:   %[[X:.*]] = arith.trunci %[[BYTES]] : vector<[32]xi8> to vector<[32]xi1>
// CHECK:   %[[OUT:.*]] = arith.extui %[[X]] : vector<[32]xi1> to vector<[32]xi8>
// CHECK:   llvm.store %[[OUT]], %{{.*}} <alignment = 1> : vector<[32]xi8>, !llvm.ptr
// CHECK: scf.for
// CHECK:   memref.load %{{.*}} : memref<128xi1>
// CHECK:   memref.store %{{.*}} : memref<128xi1>
// WIDTH4-LABEL: func.func @copy_bool
// WIDTH4: arith.trunci %{{.*}} : vector<[4]xi8> to vector<[4]xi1>
func.func @copy_bool(%a: memref<128xi1>, %b: memref<128xi1>) {
  %vl = arith.constant 128 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xi1> -> !vir.vec<?xi1>
    vir.store %x, %b[] : !vir.vec<?xi1> -> memref<128xi1>
    vector.yield
  }
  return
}

// The i1 of a comparison of f32 stored to a memref of i1.
// CHECK-LABEL: func.func @store_compare
// CHECK: %[[M:.*]] = arith.cmpf ogt, %{{.*}}, %{{.*}} : vector<[16]xf32>
// CHECK: %[[B:.*]] = arith.extui %[[M]] : vector<[16]xi1> to vector<[16]xi8>
// CHECK: llvm.store %[[B]], %{{.*}} <alignment = 1> : vector<[16]xi8>, !llvm.ptr
func.func @store_compare(%a: memref<128xf32>, %b: memref<128xi1>) {
  %vl = arith.constant 128 : index
  %zero = arith.constant 0.0 : f32
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %z = vir.broadcast %zero : f32 -> !vir.vec<?xf32>
    %m = arith.cmpf ogt, %x, %z : !vir.vec<?xf32>
    vir.store %m, %b[] : !vir.vec<?xi1> -> memref<128xi1>
    vector.yield
  }
  return
}

// A region with results, a row of a 2-d memref with an offset.
// CHECK-LABEL: func.func @results
// CHECK: llvm.load %{{.*}} <alignment = 1> : !llvm.ptr -> vector<[{{[0-9]+}}]xi8>
// CHECK: arith.trunci
func.func @results(%a: memref<128xf32>, %m: memref<4x128xi1>) -> f32 {
  %vl = arith.constant 128 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %k = vir.load %m[%c2, %c0] : memref<4x128xi1> -> !vir.vec<?xi1>
    %zv = vir.broadcast %z : f32 -> !vir.vec<?xf32>
    %y = vir.select %k, %x, %zv : !vir.vec<?xi1>, !vir.vec<?xf32>
    %s = vir.reduce %y, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// Elements that are not contiguous along the lanes (a dynamic stride): a
// strided load / store of the bytes, all lanes on.
// CHECK-LABEL: func.func @strided
// CHECK: %{{.*}}, %{{.*}}, %{{.*}}, %[[S:.*]] = memref.extract_strided_metadata
// CHECK: %[[STRIDE:.*]] = arith.index_cast %[[S]] : index to i64
// CHECK: %[[LANES:.*]] = arith.muli %{{.*}}, %{{.*}} : index
// CHECK: %[[EVL:.*]] = arith.index_cast %[[LANES]] : index to i32
// CHECK: %[[BYTES:.*]] = "llvm.intr.experimental.vp.strided.load"(%{{.*}}, %[[STRIDE]], %{{.*}}, %[[EVL]]) : (!llvm.ptr, i64, vector<[64]xi1>, i32) -> vector<[64]xi8>
// CHECK: arith.trunci %[[BYTES]] : vector<[64]xi8> to vector<[64]xi1>
// CHECK: "llvm.intr.experimental.vp.strided.store"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<[64]xi8>, !llvm.ptr, i64, vector<[64]xi1>, i32) -> ()
func.func @strided(%a: memref<?xi1, strided<[?], offset: ?>>,
                   %b: memref<?xi1, strided<[?], offset: ?>>, %vl: index) {
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<?xi1, strided<[?], offset: ?>> -> !vir.vec<?xi1>
    vir.store %x, %b[] : !vir.vec<?xi1> -> memref<?xi1, strided<[?], offset: ?>>
    vector.yield
  }
  return
}
