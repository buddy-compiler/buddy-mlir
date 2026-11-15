// RUN: buddy-opt %s -lower-vir-to-vector='vector-width=4' -cse | FileCheck %s

// A vir.reduce of a set_vl with results over several vectors of the target
// width. With reassoc, the vectors are added lane by lane first as LLVM
// splits the reduction of a vector wider than its registers (n pieces
// widened to the next power of two h, piece p with piece p + h / 2), then
// reduced once; the scalars of vl % w after. Without it, each piece is
// reduced in order.

// CHECK-LABEL: func.func @four
// CHECK: %[[A0:.*]] = vector.load %{{.*}} : memref<16xf32>, vector<4xf32>
// CHECK: %[[A1:.*]] = vector.load %{{.*}} : memref<16xf32>, vector<4xf32>
// CHECK: %[[A2:.*]] = vector.load %{{.*}} : memref<16xf32>, vector<4xf32>
// CHECK: %[[A3:.*]] = vector.load %{{.*}} : memref<16xf32>, vector<4xf32>
// CHECK: %[[S02:.*]] = arith.addf %[[A0]], %[[A2]] : vector<4xf32>
// CHECK: %[[S13:.*]] = arith.addf %[[A1]], %[[A3]] : vector<4xf32>
// CHECK: %[[S:.*]] = arith.addf %[[S02]], %[[S13]] : vector<4xf32>
// CHECK: %[[R:.*]] = vector.reduction <add>, %[[S]], %{{.*}} fastmath<reassoc> : vector<4xf32> into f32
// CHECK-NOT: vector.reduction
// CHECK: return %[[R]]
func.func @four(%a: memref<16xf32>, %z: f32) -> f32 {
  %vl = arith.constant 16 : index
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<16xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add", reassoc} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// Three vectors (h = 4): 0 with 2, then with 1; two scalars after.
// CHECK-LABEL: func.func @three_and_scalars
// CHECK: %[[A0:.*]] = vector.load %{{.*}} : memref<14xf32>, vector<4xf32>
// CHECK: %[[A1:.*]] = vector.load %{{.*}} : memref<14xf32>, vector<4xf32>
// CHECK: %[[A2:.*]] = vector.load %{{.*}} : memref<14xf32>, vector<4xf32>
// CHECK: %[[E0:.*]] = memref.load
// CHECK: %[[E1:.*]] = memref.load
// CHECK: %[[S02:.*]] = arith.maxnumf %[[A0]], %[[A2]] : vector<4xf32>
// CHECK: %[[S:.*]] = arith.maxnumf %[[S02]], %[[A1]] : vector<4xf32>
// CHECK: %[[R:.*]] = vector.reduction <maxnumf>, %[[S]], %{{.*}} fastmath<reassoc> : vector<4xf32> into f32
// CHECK: %[[R0:.*]] = arith.maxnumf %[[R]], %[[E0]] : f32
// CHECK: %[[R1:.*]] = arith.maxnumf %[[R0]], %[[E1]] : f32
// CHECK: return %[[R1]]
func.func @three_and_scalars(%a: memref<14xf32>, %z: f32) -> f32 {
  %vl = arith.constant 14 : index
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<14xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "maxnum", reassoc} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// Ordered: one reduction per piece, in order.
// CHECK-LABEL: func.func @ordered
// CHECK: %[[A0:.*]] = vector.load %{{.*}} : memref<8xf32>, vector<4xf32>
// CHECK: %[[A1:.*]] = vector.load %{{.*}} : memref<8xf32>, vector<4xf32>
// CHECK-NOT: arith.addf
// CHECK: %[[R0:.*]] = vector.reduction <add>, %[[A0]], %{{.*}} : vector<4xf32> into f32
// CHECK: %[[R1:.*]] = vector.reduction <add>, %[[A1]], %[[R0]] : vector<4xf32> into f32
// CHECK: return %[[R1]]
func.func @ordered(%a: memref<8xf32>, %z: f32) -> f32 {
  %vl = arith.constant 8 : index
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<8xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// Six vectors (h = 8): 0 + 4, 1 + 5, then (0 + 4) + 2, (1 + 5) + 3.
// CHECK-LABEL: func.func @six
// CHECK: %[[A0:.*]] = vector.load %{{.*}} : memref<24xf32>, vector<4xf32>
// CHECK: %[[A1:.*]] = vector.load %{{.*}} : memref<24xf32>, vector<4xf32>
// CHECK: %[[A2:.*]] = vector.load %{{.*}} : memref<24xf32>, vector<4xf32>
// CHECK: %[[A3:.*]] = vector.load %{{.*}} : memref<24xf32>, vector<4xf32>
// CHECK: %[[A4:.*]] = vector.load %{{.*}} : memref<24xf32>, vector<4xf32>
// CHECK: %[[A5:.*]] = vector.load %{{.*}} : memref<24xf32>, vector<4xf32>
// CHECK: %[[S04:.*]] = arith.addf %[[A0]], %[[A4]] : vector<4xf32>
// CHECK: %[[S15:.*]] = arith.addf %[[A1]], %[[A5]] : vector<4xf32>
// CHECK: %[[S042:.*]] = arith.addf %[[S04]], %[[A2]] : vector<4xf32>
// CHECK: %[[S153:.*]] = arith.addf %[[S15]], %[[A3]] : vector<4xf32>
// CHECK: %[[S:.*]] = arith.addf %[[S042]], %[[S153]] : vector<4xf32>
// CHECK: vector.reduction <add>, %[[S]], %{{.*}} fastmath<reassoc> : vector<4xf32> into f32
func.func @six(%a: memref<24xf32>, %z: f32) -> f32 {
  %vl = arith.constant 24 : index
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<24xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add", reassoc} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}
