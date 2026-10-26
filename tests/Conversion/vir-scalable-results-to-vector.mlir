// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=256' \
// RUN: | FileCheck %s --check-prefixes=CHECK,V256
// RUN: buddy-opt %s -lower-vir-to-vector='use-scalable=true rvv-vlen=1024' \
// RUN: | FileCheck %s --check-prefixes=CHECK,V1024

// A vir.set_vl with results on scalable vectors: its constant vl is covered
// by as many vector<[N]xT> pieces as at the minimum VLEN (rvv-vlen), each
// masked from vl on. Loads and stores are masked (a load passes no value
// through: the lanes that are off are not used), and a reduction sees the
// identity in those lanes (-0.0 for add, NaN for maxnum), so that the result
// is that of fixed vectors.

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
// CHECK: %[[VSCALE:.*]] = vector.vscale
// V256: %[[LANES:.*]] = arith.muli %c16, %[[VSCALE]] : index
// V256: %[[VL:.*]] = arith.constant 128 : index
// V256: %[[OFF0:.*]] = arith.muli %c0, %[[LANES]] : index
// V256: %[[REST0:.*]] = arith.subi %[[VL]], %[[OFF0]] : index
// V256: %[[M0:.*]] = vector.create_mask %[[REST0]] : vector<[16]xi1>
// V256: %[[OFF1:.*]] = arith.muli %c1, %[[LANES]] : index
// V256: %[[REST1:.*]] = arith.subi %[[VL]], %[[OFF1]] : index
// V256: %[[M1:.*]] = vector.create_mask %[[REST1]] : vector<[16]xi1>
// V256: %[[NONE:.*]] = ub.poison : vector<[16]xf32>
// V256: %[[X0:.*]] = vector.maskedload %{{.*}}, %[[M0]], %[[NONE]] : memref<128xf32>, vector<[16]xi1>, vector<[16]xf32> into vector<[16]xf32>
// V256: %[[X1:.*]] = vector.maskedload %{{.*}}, %[[M1]], %{{.*}}
// V256: %[[P0:.*]] = arith.mulf
// V256: %[[P1:.*]] = arith.mulf
// V256: vector.maskedstore %{{.*}}, %[[M0]], %[[P0]]
// V256: vector.maskedstore %{{.*}}, %[[M1]], %[[P1]]
// V256: %[[ID:.*]] = arith.constant dense<-0.000000e+00> : vector<[16]xf32>
// V256: %[[S0:.*]] = arith.select %[[M0]], %[[P0]], %[[ID]]
// V256: %[[R0:.*]] = vector.reduction <add>, %[[S0]], %{{.*}} : vector<[16]xf32> into f32
// V256: %[[S1:.*]] = arith.select %[[M1]], %[[P1]], %{{.*}}
// V256: %[[R1:.*]] = vector.reduction <add>, %[[S1]], %[[R0]]
// V256: return %[[R1]] : f32
// V1024: arith.muli %c8, %[[VSCALE]] : index
// V1024: vector.create_mask %{{.*}} : vector<[8]xi1>
// V1024-NOT: vector.create_mask
// V1024: vector.maskedload {{.*}} vector<[8]xf32>
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
// CHECK: %[[NAN:.*]] = arith.constant dense<0x7FC00000> : vector<[{{8|16}}]xf32>
// CHECK: %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %[[NAN]]
// CHECK: vector.reduction <maxnumf>, %[[S]]

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
// CHECK: vector.maskedload
