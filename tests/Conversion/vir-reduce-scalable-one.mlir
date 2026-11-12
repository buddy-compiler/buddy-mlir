// RUN: buddy-opt %s \
// RUN:     -lower-vir-to-vector='rvv-vlen=1024 use-scalable=true' -canonicalize \
// RUN: | FileCheck %s

// At VLEN 1024, the 16 lanes of these regions are one vector<[1]xf32>. Its
// reduction must reduce all vscale lanes: the canonicalization of
// vector.reduction takes vector<[1]xT> for one element and keeps lane 0
// only, so the reduction is built from the LLVM intrinsics.

// CHECK-LABEL: func.func @max
// CHECK: %[[X:.*]] = vector.load %{{.*}} : memref<16xf32>, vector<[1]xf32>
// CHECK-NOT: vector.extract
// CHECK: %[[M:.*]] = llvm.intr.vector.reduce.fmax(%[[X]])
// CHECK: arith.maxnumf %[[M]], %{{.*}} : f32
func.func @max(%a: memref<16xf32>, %s: memref<f32>) {
  %vl = arith.constant 16 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<16xf32> -> !vir.vec<?xf32>
    %acc = memref.load %s[] : memref<f32>
    %r = vir.reduce %x, %acc {kind = "maxnum"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %r, %s[] : memref<f32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @sum
// CHECK: %[[X:.*]] = vector.load %{{.*}} : memref<16xf32>, vector<[1]xf32>
// CHECK-NOT: vector.extract
// CHECK: llvm.intr.vector.reduce.fadd(%{{.*}}, %[[X]])
func.func @sum(%a: memref<16xf32>, %s: memref<f32>) {
  %vl = arith.constant 16 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<16xf32> -> !vir.vec<?xf32>
    %acc = memref.load %s[] : memref<f32>
    %r = vir.reduce %x, %acc {kind = "add"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %r, %s[] : memref<f32>
    vector.yield
  }
  return
}

// With results: the region of 16 lanes is one vector<[1]xf32> too.
// CHECK-LABEL: func.func @results
// CHECK-NOT: vector.extract
// CHECK: llvm.intr.vector.reduce.fadd(%{{.*}}, %{{.*}}) fastmath<reassoc>
func.func @results(%a: memref<16xf32>, %z: f32) -> f32 {
  %vl = arith.constant 16 : index
  %r = vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<16xf32> -> !vir.vec<?xf32>
    %s = vir.reduce %x, %z {kind = "add", reassoc} : !vir.vec<?xf32>, f32 -> f32
    vector.yield %s : f32
  } -> f32
  return %r : f32
}

// The other kinds too: an integer one reduces the lanes with its intrinsic,
// then combines the accumulator; a float product starts from it, ordered.
// CHECK-LABEL: func.func @min_i32
// CHECK: %[[X:.*]] = vector.load %{{.*}} : memref<16xi32>, vector<[1]xi32>
// CHECK-NOT: vector.extract
// CHECK: %[[M:.*]] = "llvm.intr.vector.reduce.smin"(%[[X]])
// CHECK: arith.minsi %[[M]], %{{.*}} : i32
func.func @min_i32(%a: memref<16xi32>, %s: memref<i32>) {
  %vl = arith.constant 16 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<16xi32> -> !vir.vec<?xi32>
    %acc = memref.load %s[] : memref<i32>
    %r = vir.reduce %x, %acc {kind = "minsi"} : !vir.vec<?xi32>, i32 -> i32
    memref.store %r, %s[] : memref<i32>
    vector.yield
  }
  return
}

// CHECK-LABEL: func.func @prod
// CHECK: %[[X:.*]] = vector.load %{{.*}} : memref<16xf32>, vector<[1]xf32>
// CHECK-NOT: vector.extract
// CHECK: llvm.intr.vector.reduce.fmul(%{{.*}}, %[[X]])
func.func @prod(%a: memref<16xf32>, %s: memref<f32>) {
  %vl = arith.constant 16 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<16xf32> -> !vir.vec<?xf32>
    %acc = memref.load %s[] : memref<f32>
    %r = vir.reduce %x, %acc {kind = "mul"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %r, %s[] : memref<f32>
    vector.yield
  }
  return
}

// minimumnumf: LLVM has no scalable fminimumnum reduction on RVV, the NaN
// lanes become +inf for an fminimum reduction (of all lanes too).
// CHECK-LABEL: func.func @min_num
// CHECK: %[[X:.*]] = vector.load %{{.*}} : memref<16xf32>, vector<[1]xf32>
// CHECK-NOT: vector.extract
// CHECK: %[[N:.*]] = arith.cmpf uno, %[[X]], %[[X]] : vector<[1]xf32>
// CHECK: %[[L:.*]] = arith.select %[[N]], %{{.*}}, %[[X]] : vector<[1]xi1>, vector<[1]xf32>
// CHECK: llvm.intr.vector.reduce.fminimum(%[[L]])
// CHECK: "llvm.intr.vector.reduce.and"(%[[N]])
// CHECK: arith.minimumnumf %{{.*}}, %{{.*}} : f32
func.func @min_num(%a: memref<16xf32>, %s: memref<f32>) {
  %vl = arith.constant 16 : index
  vir.set_vl %vl : index {
    %x = vir.load %a[] : memref<16xf32> -> !vir.vec<?xf32>
    %acc = memref.load %s[] : memref<f32>
    %r = vir.reduce %x, %acc {kind = "minimumnumf"} : !vir.vec<?xf32>, f32 -> f32
    memref.store %r, %s[] : memref<f32>
    vector.yield
  }
  return
}
