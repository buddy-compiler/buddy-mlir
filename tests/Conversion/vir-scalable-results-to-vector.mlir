// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN: | FileCheck %s --check-prefixes=CHECK,V256
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=1024' \
// RUN: | FileCheck %s --check-prefixes=CHECK,V1024

// A vir.set_vl with results on scalable vectors, for the VLEN rvv-vlen
// (vscale = rvv-vlen / 64): its constant vl is covered by whole
// vector<[N]xT> pieces of N * vscale lanes, then the lanes left as scalars,
// as with fixed vectors. No lane is off: no masks.

// LMUL 8: at VLEN 256, vector<[16]xf32> holds 64 lanes, two pieces for a vl
// of 128; at VLEN 1024, the clamp to vl gives vector<[8]xf32> of 128 lanes,
// one piece.
func.func @dot(%a: memref<128xf32>, %b: memref<128xf32>) -> f32 {
  %vl = arith.constant 128 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %y = vir.load %b[] : memref<128xf32> -> !vir.vec<?xf32>
    %m = arith.mulf %x, %y : !vir.vec<?xf32>
    vir.store %m, %b[] : !vir.vec<?xf32> -> memref<128xf32>
    %s = vir.reduce %m, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @dot
// CHECK-NOT: mask
// V256: %[[X0:.*]] = vector.load %{{.*}} : memref<128xf32>, vector<[16]xf32>
// V256: %[[AT1:.*]] = arith.addi %{{.*}}, %c64{{.*}} : index
// V256: %[[X1:.*]] = vector.load %{{.*}}[%[[AT1]]] : memref<128xf32>, vector<[16]xf32>
// V256: %[[Y0:.*]] = vector.load
// V256: %[[Y1:.*]] = vector.load
// V256: %[[P0:.*]] = arith.mulf %[[X0]], %[[Y0]]
// V256: %[[P1:.*]] = arith.mulf %[[X1]], %[[Y1]]
// V256: vector.store %[[P0]], %{{.*}} : memref<128xf32>, vector<[16]xf32>
// V256: vector.store %[[P1]], %{{.*}} : memref<128xf32>, vector<[16]xf32>
// V256: %[[R0:.*]] = vector.reduction <add>, %[[P0]], %{{.*}} : vector<[16]xf32> into f32
// V256: %[[R1:.*]] = vector.reduction <add>, %[[P1]], %[[R0]] : vector<[16]xf32> into f32
// V256: return %[[R1]] : f32
// V1024: vector.load {{.*}} vector<[8]xf32>
// V1024: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<[8]xf32> into f32
// V1024-NOT: vector.reduction
// V1024: return

func.func @max(%a: memref<128xf32>) -> f32 {
  %vl = arith.constant 128 : index
  %init = arith.constant 0xFF800000 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<128xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %init {kind = "maxnum"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @max
// CHECK-NOT: select
// CHECK: vector.reduction <maxnumf>, %{{.*}}, %{{.*}} : vector<[{{8|16}}]xf32> into f32

// A loop carries the pieces as scalable vectors.
func.func @loop(%k: memref<3x128xf32>) -> f32 {
  %vl = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %init = vir.broadcast %z : f32 -> !vir.vec<?xf32>
    %acc = scf.for %i = %c0 to %c3 step %c1 iter_args(%a = %init) -> (!vir.vec<?xf32>) {
      %kv = vir.load %k[%i, %c0] : memref<3x128xf32> -> !vir.vec<?xf32>
      %n = arith.addf %a, %kv : !vir.vec<?xf32>
      scf.yield %n : !vir.vec<?xf32>
    }
    %s = vir.reduce %acc, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @loop
// V256: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (vector<[16]xf32>, vector<[16]xf32>) {
// V1024: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<[8]xf32>) {
// CHECK: vector.load %{{.*}} : memref<3x128xf32>, vector<[{{8|16}}]xf32>

// A vl of 72: a vector of 64 lanes (vector<[16]xf32> at VLEN 256,
// vector<[4]xf32> at VLEN 1024, clamped to vl), then 8 scalars.
func.func @tail(%a: memref<72xf32>, %o: memref<72xf32>) -> f32 {
  %vl = arith.constant 72 : index
  %z = arith.constant 0.0 : f32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<72xf32> -> !vir.vec<?xf32>
    %y = arith.mulf %x, %x : !vir.vec<?xf32>
    vir.store %y, %o[] : !vir.vec<?xf32> -> memref<72xf32>
    %s = vir.reduce %y, %z {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @tail
// CHECK-NOT: mask
// V256: vector.load %{{.*}} : memref<72xf32>, vector<[16]xf32>
// V1024: vector.load %{{.*}} : memref<72xf32>, vector<[4]xf32>
// CHECK-COUNT-8: memref.load %{{.*}} : memref<72xf32>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<[{{4|16}}]xf32>
// CHECK-COUNT-8: arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK: vector.store %{{.*}} : memref<72xf32>, vector<[{{4|16}}]xf32>
// CHECK-COUNT-8: memref.store %{{.*}} : memref<72xf32>
// CHECK: vector.reduction <add>
// CHECK-COUNT-8: arith.addf %{{.*}}, %{{.*}} : f32

// A vl of 3 is fewer lanes than the smallest vector<[1]xi32> (4 lanes at
// VLEN 256): scalars only.
func.func @short(%a: memref<3xi32>, %o: memref<3xi32>) -> i32 {
  %vl = arith.constant 3 : index
  %c0 = arith.constant 0 : i32
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<3xi32> -> !vir.vec<?xi32>
    %y = arith.addi %x, %x : !vir.vec<?xi32>
    vir.store %y, %o[] : !vir.vec<?xi32> -> memref<3xi32>
    vector.yield %c0 : i32
  } -> i32
  return %r : i32
}

// CHECK-LABEL: func.func @short
// CHECK-NOT: vector
// CHECK-COUNT-3: arith.addi %{{.*}}, %{{.*}} : i32
