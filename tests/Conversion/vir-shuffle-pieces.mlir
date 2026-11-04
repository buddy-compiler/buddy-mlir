// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN: | FileCheck %s

// A vir.shuffle on scalable vectors of several pieces: every lane is known
// at VLEN 256 (vscale 4), so a piece of the result is made of runs of lanes
// from pieces of the operands.

// A row of 64 f32 is one vector<[16]xf32> (LMUL 8, 64 lanes): the rows swap
// as whole pieces, no instruction.
func.func @rows64(%a: memref<2x64xf32>, %o: memref<2x64xf32>) {
  %vl = arith.constant 64 : index
  %z = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x64xf32> -> !vir.vec<2x?xf32>
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xf32>
    vir.store %y, %o[] : !vir.vec<2x?xf32> -> memref<2x64xf32>
    vector.yield %z : i32
  } -> i32
  return
}

// CHECK-LABEL: func.func @rows64
// CHECK-NOT: splice
// CHECK-NOT: scalable
// CHECK: %[[X0:.*]] = vector.load %{{.*}} : memref<128xf32>, vector<[16]xf32>
// CHECK: %[[X1:.*]] = vector.load %{{.*}} : memref<128xf32>, vector<[16]xf32>
// CHECK: vector.store %[[X1]]
// CHECK: vector.store %[[X0]]

// 48 lanes are three vector<[4]xf32> of 16 lanes (whole vectors only: no
// lanes left as scalars); a row of 24 lanes is 6 blocks of vscale lanes, so
// the runs are parts vector<[2]xf32> taken and put in place.
func.func @rows24(%a: memref<2x24xf32>, %o: memref<2x24xf32>) {
  %vl = arith.constant 24 : index
  %z = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x24xf32> -> !vir.vec<2x?xf32>
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xf32>
    vir.store %y, %o[] : !vir.vec<2x?xf32> -> memref<2x24xf32>
    vector.yield %z : i32
  } -> i32
  return
}

// CHECK-LABEL: func.func @rows24
// CHECK-NOT: splice
// CHECK: %[[X0:.*]] = vector.load %{{.*}} : memref<48xf32>, vector<[4]xf32>
// CHECK: %[[X1:.*]] = vector.load %{{.*}} : memref<48xf32>, vector<[4]xf32>
// CHECK: %[[X2:.*]] = vector.load %{{.*}} : memref<48xf32>, vector<[4]xf32>
// CHECK: %[[A:.*]] = vector.scalable.extract %[[X1]][2] : vector<[2]xf32> from vector<[4]xf32>
// CHECK: %[[P:.*]] = vector.scalable.insert %[[A]], %{{.*}}[0] : vector<[2]xf32> into vector<[4]xf32>
// CHECK: %[[B:.*]] = vector.scalable.extract %[[X2]][0] : vector<[2]xf32> from vector<[4]xf32>
// CHECK: vector.scalable.insert %[[B]], %[[P]][2] : vector<[2]xf32> into vector<[4]xf32>

// Rows of 11 lanes in pieces of 4 (vector<[1]xf32>): the runs are slid into
// place and selected from their first lane on.
func.func @rows11(%a: memref<4x11xf32>, %b: memref<4x11xf32>, %o: memref<4x11xf32>) {
  %vl = arith.constant 11 : index
  %z = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<4x11xf32> -> !vir.vec<4x?xf32>
    %y = vir.load %b[] : memref<4x11xf32> -> !vir.vec<4x?xf32>
    %w = vir.shuffle %x, %y [1, 6, 0, 3] : !vir.vec<4x?xf32>
    vir.store %w, %o[] : !vir.vec<4x?xf32> -> memref<4x11xf32>
    vector.yield %z : i32
  } -> i32
  return
}

// CHECK-LABEL: func.func @rows11
// CHECK: vector.load %{{.*}} : memref<44xf32>, vector<[1]xf32>
// lanes 0..3 of the result: lanes 11..14 of a, lanes 3 of piece 2 on
// CHECK: %[[S:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%{{.*}}, %{{.*}}, %c3_i32) : (vector<[1]xf32>, vector<[1]xf32>, i32) -> vector<[1]xf32>
// CHECK: %[[T:.*]] = llvm.call_intrinsic "llvm.vector.splice.right"(%{{.*}}, %{{.*}}, %c1_i32) : (vector<[1]xf32>, vector<[1]xf32>, i32) -> vector<[1]xf32>
// CHECK: %[[M:.*]] = vector.create_mask %c1{{.*}} : vector<[1]xi1>
// CHECK: arith.select %[[M]], %[[S]], %[[T]] : vector<[1]xi1>, vector<[1]xf32>

// 6 lanes are a vector<[1]xf32> of 4 lanes and 2 scalars: a scalar of the
// result is a lane slid down to lane 0, a scalar of an operand a broadcast
// (to vector<[2]xf32>, then its first half: see broadcastTo) selected from
// its lane on.
func.func @tail(%a: memref<2x3xf32>, %o: memref<2x3xf32>) {
  %vl = arith.constant 3 : index
  %z = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<2x3xf32> -> !vir.vec<2x?xf32>
    %y = vir.shuffle %x, %x [1, 0] : !vir.vec<2x?xf32>
    vir.store %y, %o[] : !vir.vec<2x?xf32> -> memref<2x3xf32>
    vector.yield %z : i32
  } -> i32
  return
}

// CHECK-LABEL: func.func @tail
// CHECK: %[[X:.*]] = vector.load %{{.*}} : memref<6xf32>, vector<[1]xf32>
// CHECK: %[[X4:.*]] = memref.load
// CHECK: %[[X5:.*]] = memref.load
// CHECK: %[[S:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%[[X]], %{{.*}}, %c3_i32)
// CHECK: %[[W:.*]] = vector.broadcast %[[X4]] : f32 to vector<[2]xf32>
// CHECK: %[[B:.*]] = vector.scalable.extract %[[W]][0] : vector<[1]xf32> from vector<[2]xf32>
// CHECK: %[[M:.*]] = vector.create_mask %c1{{.*}} : vector<[1]xi1>
// CHECK: arith.select %[[M]], %[[S]], %[[B]]
// CHECK: %[[L:.*]] = llvm.call_intrinsic "llvm.vector.splice.left"(%[[X]], %{{.*}}, %c1_i32)
// CHECK: %[[E:.*]] = vector.extract %[[L]][0] : f32 from vector<[1]xf32>
// CHECK: memref.store %[[E]]
