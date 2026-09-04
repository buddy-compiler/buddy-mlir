// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir | FileCheck %s
// RUN: buddy-opt %s -split-input-file -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse | FileCheck %s --check-prefix=CHECK-VEC-REDUCE

// -----
// CASE: linalg-reduce-to-vir-scalar-addf-f32.mlir
func.func @reduce_addf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_addf
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_addf
// CHECK-VEC-REDUCE: vector.reduction <add>
// CHECK-VEC-REDUCE: arith.addf

// -----
// CASE: linalg-reduce-to-vir-scalar-addi-i32.mlir
func.func @reduce_addi(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %res = arith.addi %in, %init : i32
      linalg.yield %res : i32
    }
  return
}

// CHECK-LABEL: func.func @reduce_addi
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_addi
// CHECK-VEC-REDUCE: vector.reduction <add>
// CHECK-VEC-REDUCE: arith.addi

// -----
// CASE: linalg-reduce-to-vir-scalar-andi-i1.mlir
func.func @reduce_andi(%arg0: memref<16xi1>, %arg1: memref<i1>) {
  linalg.reduce ins(%arg0 : memref<16xi1>) outs(%arg1 : memref<i1>) dimensions = [0]
    (%in: i1, %init: i1) {
      %res = arith.andi %in, %init : i1
      linalg.yield %res : i1
    }
  return
}

// CHECK-LABEL: func.func @reduce_andi
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_andi
// CHECK-VEC-REDUCE: vector.reduction <and>
// CHECK-VEC-REDUCE: arith.andi

// -----
// CASE: linalg-reduce-to-vir-scalar-maximumf-f32.mlir
func.func @reduce_maximumf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %res = arith.maximumf %in, %init : f32
      linalg.yield %res : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_maximumf
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_maximumf
// CHECK-VEC-REDUCE: vector.reduction <maximumf>
// CHECK-VEC-REDUCE: arith.maximumf

// -----
// CASE: linalg-reduce-to-vir-scalar-maxnumf-f32.mlir
func.func @reduce_maxnumf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %max = arith.maxnumf %in, %init : f32
      linalg.yield %max : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_maxnumf
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_maxnumf
// CHECK-VEC-REDUCE: vector.reduction <maxnumf>
// CHECK-VEC-REDUCE: arith.maxnumf

// -----
// CASE: linalg-reduce-to-vir-scalar-maxsi-i32.mlir
func.func @reduce_maxsi(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %res = arith.maxsi %in, %init : i32
      linalg.yield %res : i32
    }
  return
}

// CHECK-LABEL: func.func @reduce_maxsi
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_maxsi
// CHECK-VEC-REDUCE: vector.reduction <maxsi>
// CHECK-VEC-REDUCE: arith.maxsi

// -----
// CASE: linalg-reduce-to-vir-scalar-maxui-i32.mlir
func.func @reduce_maxui(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %res = arith.maxui %in, %init : i32
      linalg.yield %res : i32
    }
  return
}

// CHECK-LABEL: func.func @reduce_maxui
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_maxui
// CHECK-VEC-REDUCE: vector.reduction <maxui>
// CHECK-VEC-REDUCE: arith.maxui

// -----
// CASE: linalg-reduce-to-vir-scalar-minimumf-f32.mlir
func.func @reduce_minimumf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %res = arith.minimumf %in, %init : f32
      linalg.yield %res : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_minimumf
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_minimumf
// CHECK-VEC-REDUCE: vector.reduction <minimumf>
// CHECK-VEC-REDUCE: arith.minimumf

// -----
// CASE: linalg-reduce-to-vir-scalar-minnumf-f32.mlir
func.func @reduce_minnumf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %res = arith.minnumf %in, %init : f32
      linalg.yield %res : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_minnumf
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_minnumf
// CHECK-VEC-REDUCE: vector.reduction <minnumf>
// CHECK-VEC-REDUCE: arith.minnumf

// -----
// CASE: linalg-reduce-to-vir-scalar-minsi-i32.mlir
func.func @reduce_minsi(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %res = arith.minsi %in, %init : i32
      linalg.yield %res : i32
    }
  return
}

// CHECK-LABEL: func.func @reduce_minsi
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_minsi
// CHECK-VEC-REDUCE: vector.reduction <minsi>
// CHECK-VEC-REDUCE: arith.minsi

// -----
// CASE: linalg-reduce-to-vir-scalar-minui-i32.mlir
func.func @reduce_minui(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %res = arith.minui %in, %init : i32
      linalg.yield %res : i32
    }
  return
}

// CHECK-LABEL: func.func @reduce_minui
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_minui
// CHECK-VEC-REDUCE: vector.reduction <minui>
// CHECK-VEC-REDUCE: arith.minui

// -----
// CASE: linalg-reduce-to-vir-scalar-mulf-f32.mlir
func.func @reduce_mulf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %res = arith.mulf %in, %init : f32
      linalg.yield %res : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_mulf
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_mulf
// CHECK-VEC-REDUCE: vector.reduction <mul>
// CHECK-VEC-REDUCE: arith.mulf

// -----
// CASE: linalg-reduce-to-vir-scalar-ori-i32.mlir
func.func @reduce_ori(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %res = arith.ori %in, %init : i32
      linalg.yield %res : i32
    }
  return
}

// CHECK-LABEL: func.func @reduce_ori
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_ori
// CHECK-VEC-REDUCE: vector.reduction <or>
// CHECK-VEC-REDUCE: arith.ori

// -----
// CASE: linalg-reduce-to-vir-scalar-xori-i32.mlir
func.func @reduce_xori(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %res = arith.xori %in, %init : i32
      linalg.yield %res : i32
    }
  return
}

// CHECK-LABEL: func.func @reduce_xori
// CHECK-NOT: linalg.reduce
// CHECK: vir.set_vl
// CHECK: vir.load %arg0
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_xori
// CHECK-VEC-REDUCE: vector.reduction <xor>
// CHECK-VEC-REDUCE: arith.xori

// -----
// CASE: linalg-reduce-to-vir-2d-to-1d-addf-f32.mlir
func.func @reduce_2d_to_1d_addf(%arg0: memref<4x8xf32>, %arg1: memref<4xf32>) {
  linalg.reduce ins(%arg0 : memref<4x8xf32>) outs(%arg1 : memref<4xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_2d_to_1d_addf
// CHECK-NOT: linalg.reduce
// CHECK: scf.for %[[I:[^ ]+]] =
// CHECK: %[[INIT:.+]] = memref.load %arg1[%[[I]]]
// CHECK: vir.set_vl
// CHECK: vir.load %arg0[%[[I]]
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_2d_to_1d_addf
// CHECK-VEC-REDUCE: vector.reduction <add>
// CHECK-VEC-REDUCE: arith.addf

// -----
// CASE: linalg-reduce-to-vir-2d-to-1d-maxnumf-f32.mlir
func.func @reduce_2d_to_1d_maxnumf(%arg0: memref<4x8xf32>, %arg1: memref<4xf32>) {
  linalg.reduce ins(%arg0 : memref<4x8xf32>) outs(%arg1 : memref<4xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %max = arith.maxnumf %in, %init : f32
      linalg.yield %max : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_2d_to_1d_maxnumf
// CHECK-NOT: linalg.reduce
// CHECK: scf.for %[[I:[^ ]+]] =
// CHECK: %[[INIT:.+]] = memref.load %arg1[%[[I]]]
// CHECK: vir.set_vl
// CHECK: vir.load %arg0[%[[I]]
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_2d_to_1d_maxnumf
// CHECK-VEC-REDUCE: vector.reduction <maxnumf>
// CHECK-VEC-REDUCE: arith.maxnumf

// -----
// CASE: linalg-reduce-to-vir-2d-dim0-addf-f32.mlir
func.func @reduce_2d_dim0_addf(%arg0: memref<4x8xf32>, %arg1: memref<8xf32>) {
  linalg.reduce ins(%arg0 : memref<4x8xf32>) outs(%arg1 : memref<8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_2d_dim0_addf
// CHECK-NOT: linalg.reduce
// CHECK-NOT: memref.alloca_scope
// CHECK: memref.transpose %arg0 {{.*}} : memref<4x8xf32> to memref<8x4xf32, {{.*}}>
// CHECK: %[[SCRATCH:.+]] = memref.alloc() : memref<8x4xf32>
// CHECK-NEXT: memref.copy %transpose, %[[SCRATCH]] : memref<8x4xf32, {{.*}}> to memref<8x4xf32>
// CHECK: scf.for %[[J:[^ ]+]] =
// CHECK: %[[INIT:.+]] = memref.load %arg1[%[[J]]]
// CHECK: vir.set_vl
// CHECK: vir.load {{.*}}[%[[J]], {{[^]]+}}]
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK: %[[FINAL:.+]] = memref.load %alloca[] : memref<f32>
// CHECK-NEXT: memref.store %[[FINAL]], %arg1[%[[J]]]
// CHECK-NEXT: }
// CHECK-NEXT: memref.dealloc %[[SCRATCH]] : memref<8x4xf32>
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_2d_dim0_addf
// CHECK-VEC-REDUCE: memref.transpose %arg0 {{.*}} : memref<4x8xf32> to memref<8x4xf32, {{.*}}>
// CHECK-VEC-REDUCE: vector.reduction <add>
// CHECK-VEC-REDUCE: arith.addf

// -----
// CASE: linalg-reduce-to-vir-2d-dim0-maxnumf-f32.mlir
func.func @reduce_2d_dim0_maxnumf(%arg0: memref<4x8xf32>, %arg1: memref<8xf32>) {
  linalg.reduce ins(%arg0 : memref<4x8xf32>) outs(%arg1 : memref<8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %max = arith.maxnumf %in, %init : f32
      linalg.yield %max : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_2d_dim0_maxnumf
// CHECK-NOT: linalg.reduce
// CHECK-NOT: memref.alloca_scope
// CHECK: memref.transpose %arg0 {{.*}} : memref<4x8xf32> to memref<8x4xf32, {{.*}}>
// CHECK: %[[SCRATCH:.+]] = memref.alloc() : memref<8x4xf32>
// CHECK-NEXT: memref.copy %transpose, %[[SCRATCH]] : memref<8x4xf32, {{.*}}> to memref<8x4xf32>
// CHECK: scf.for %[[J:[^ ]+]] =
// CHECK: %[[INIT:.+]] = memref.load %arg1[%[[J]]]
// CHECK: vir.set_vl
// CHECK: vir.load {{.*}}[%[[J]], {{[^]]+}}]
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK: %[[FINAL:.+]] = memref.load %alloca[] : memref<f32>
// CHECK-NEXT: memref.store %[[FINAL]], %arg1[%[[J]]]
// CHECK-NEXT: }
// CHECK-NEXT: memref.dealloc %[[SCRATCH]] : memref<8x4xf32>
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_2d_dim0_maxnumf
// CHECK-VEC-REDUCE: memref.transpose %arg0 {{.*}} : memref<4x8xf32> to memref<8x4xf32, {{.*}}>
// CHECK-VEC-REDUCE: vector.reduction <maxnumf>
// CHECK-VEC-REDUCE: arith.maxnumf

// -----
// CASE: linalg-reduce-to-vir-3d-dim0-addf-f32.mlir
func.func @reduce_3d_dim0_addf(%arg0: memref<8x4x8xf32>, %arg1: memref<4x8xf32>) {
  linalg.reduce ins(%arg0 : memref<8x4x8xf32>) outs(%arg1 : memref<4x8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_3d_dim0_addf
// CHECK-NOT: linalg.reduce
// CHECK: memref.transpose %arg0 {{.*}} : memref<8x4x8xf32> to memref<4x8x8xf32, {{.*}}>
// CHECK: memref.alloc{{.*}} : memref<4x8x8xf32>
// CHECK: scf.for %[[J:[^ ]+]] =
// CHECK: scf.for %[[K:[^ ]+]] =
// CHECK: %[[INIT:.+]] = memref.load %arg1[%[[J]], %[[K]]]
// CHECK: vir.set_vl
// CHECK: vir.load {{.*}}[%[[J]], %[[K]], {{[^]]+}}]
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK: memref.dealloc
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_3d_dim0_addf
// CHECK-VEC-REDUCE: memref.transpose %arg0 {{.*}} : memref<8x4x8xf32> to memref<4x8x8xf32, {{.*}}>
// CHECK-VEC-REDUCE: vector.reduction <add>
// CHECK-VEC-REDUCE: arith.addf

// -----
// CASE: linalg-reduce-to-vir-3d-dim0-maxnumf-f32.mlir
func.func @reduce_3d_dim0_maxnumf(%arg0: memref<8x4x8xf32>, %arg1: memref<4x8xf32>) {
  linalg.reduce ins(%arg0 : memref<8x4x8xf32>) outs(%arg1 : memref<4x8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %max = arith.maxnumf %in, %init : f32
      linalg.yield %max : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_3d_dim0_maxnumf
// CHECK-NOT: linalg.reduce
// CHECK: memref.transpose %arg0 {{.*}} : memref<8x4x8xf32> to memref<4x8x8xf32, {{.*}}>
// CHECK: memref.alloc{{.*}} : memref<4x8x8xf32>
// CHECK: scf.for %[[J:[^ ]+]] =
// CHECK: scf.for %[[K:[^ ]+]] =
// CHECK: %[[INIT:.+]] = memref.load %arg1[%[[J]], %[[K]]]
// CHECK: vir.set_vl
// CHECK: vir.load {{.*}}[%[[J]], %[[K]], {{[^]]+}}]
// CHECK: vir.reduce
// CHECK: memref.store
// CHECK: memref.dealloc
// CHECK-VEC-REDUCE-LABEL: func.func @reduce_3d_dim0_maxnumf
// CHECK-VEC-REDUCE: memref.transpose %arg0 {{.*}} : memref<8x4x8xf32> to memref<4x8x8xf32, {{.*}}>
// CHECK-VEC-REDUCE: vector.reduction <maxnumf>
// CHECK-VEC-REDUCE: arith.maxnumf

// -----
// CASE: linalg-reduce-to-vir-2d-to-1d-dim0-dynamic-shape.mlir
func.func @reduce_2d_to_1d_dim0_dynamic_shape(%arg0: memref<?x?xf32>, %arg1: memref<?xf32>) {
  linalg.reduce ins(%arg0 : memref<?x?xf32>) outs(%arg1 : memref<?xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_2d_to_1d_dim0_dynamic_shape
// CHECK-NOT: linalg.reduce
// CHECK: scf.for
// CHECK-NOT: linalg.reduce
// CHECK: return

// -----
// CASE: linalg-reduce-to-vir-2d-to-1d-dim0-dynamic-m.mlir
func.func @reduce_2d_to_1d_dim0_dynamic_m(%arg0: memref<?x8xf32>, %arg1: memref<8xf32>) {
  linalg.reduce ins(%arg0 : memref<?x8xf32>) outs(%arg1 : memref<8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_2d_to_1d_dim0_dynamic_m
// CHECK-NOT: linalg.reduce
// CHECK: scf.for
// CHECK-NOT: linalg.reduce
// CHECK: return

// -----
// CASE: linalg-reduce-to-vir-2d-to-1d-dim0-dynamic-n.mlir
func.func @reduce_2d_to_1d_dim0_dynamic_n(%arg0: memref<4x?xf32>, %arg1: memref<?xf32>) {
  linalg.reduce ins(%arg0 : memref<4x?xf32>) outs(%arg1 : memref<?xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_2d_to_1d_dim0_dynamic_n
// CHECK-NOT: linalg.reduce
// CHECK: scf.for
// CHECK-NOT: linalg.reduce
// CHECK: return

// -----
// CASE: linalg-reduce-to-vir-1d-to-0d-i32.mlir
func.func @reduce_1d_to_0d_i32(%arg0: memref<16xi32>, %arg1: memref<i32>) {
  linalg.reduce ins(%arg0 : memref<16xi32>) outs(%arg1 : memref<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %sum = arith.addi %in, %init : i32
      linalg.yield %sum : i32
    }
  return
}

// CHECK-LABEL: func.func @reduce_1d_to_0d_i32
// CHECK-NOT: linalg.reduce
// CHECK: vir.reduce
// CHECK-NOT: linalg.reduce
// CHECK: return

// -----
// CASE: linalg-reduce-to-vir-1d-to-0d-mulf.mlir
func.func @reduce_1d_to_0d_mulf(%arg0: memref<16xf32>, %arg1: memref<f32>) {
  linalg.reduce ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<f32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %mul = arith.mulf %in, %init : f32
      linalg.yield %mul : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_1d_to_0d_mulf
// CHECK-NOT: linalg.reduce
// CHECK: vir.reduce
// CHECK-NOT: linalg.reduce
// CHECK: return

// -----
// CASE: linalg-reduce-to-vir-3d-to-2d.mlir
func.func @reduce_3d_to_2d(%arg0: memref<2x4x8xf32>, %arg1: memref<2x4xf32>) {
  linalg.reduce ins(%arg0 : memref<2x4x8xf32>) outs(%arg1 : memref<2x4xf32>) dimensions = [2]
    (%in: f32, %init0: f32) {
      %sum = arith.addf %in, %init0 : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_3d_to_2d
// CHECK-NOT: linalg.reduce
// CHECK: scf.for
// CHECK-NOT: linalg.reduce
// CHECK: return

// -----
// CASE: linalg-reduce-to-vir-3d-to-2d-dim0-dynamic-m.mlir
func.func @reduce_3d_to_2d_dim0_dynamic_m(%arg0: memref<?x4x8xf32>, %arg1: memref<4x8xf32>) {
  linalg.reduce ins(%arg0 : memref<?x4x8xf32>) outs(%arg1 : memref<4x8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_3d_to_2d_dim0_dynamic_m
// CHECK-NOT: linalg.reduce
// CHECK: scf.for
// CHECK-NOT: linalg.reduce
// CHECK: return

// -----
// CASE: linalg-reduce-to-vir-3d-to-2d-dim0-dynamic-n.mlir
func.func @reduce_3d_to_2d_dim0_dynamic_n(%arg0: memref<2x?x8xf32>, %arg1: memref<?x8xf32>) {
  linalg.reduce ins(%arg0 : memref<2x?x8xf32>) outs(%arg1 : memref<?x8xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_3d_to_2d_dim0_dynamic_n
// CHECK-NOT: linalg.reduce
// CHECK: scf.for
// CHECK-NOT: linalg.reduce
// CHECK: return

// -----
// CASE: linalg-reduce-to-vir-3d-to-2d-dim0-dynamic-k.mlir
func.func @reduce_3d_to_2d_dim0_dynamic_k(%arg0: memref<2x4x?xf32>, %arg1: memref<4x?xf32>) {
  linalg.reduce ins(%arg0 : memref<2x4x?xf32>) outs(%arg1 : memref<4x?xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
  return
}

// CHECK-LABEL: func.func @reduce_3d_to_2d_dim0_dynamic_k
// CHECK-NOT: linalg.reduce
// CHECK: scf.for
// CHECK-NOT: linalg.reduce
// CHECK: return

// -----
// CASE: linalg-reduce-native-scalar-loop-multi-output.mlir
func.func @reduce_argmax_pair(%values: memref<16xf32>, %indices: memref<16xi32>,
                              %out_value: memref<f32>, %out_index: memref<i32>) {
  linalg.reduce ins(%values, %indices : memref<16xf32>, memref<16xi32>)
                outs(%out_value, %out_index : memref<f32>, memref<i32>)
                dimensions = [0]
    (%value: f32, %index: i32, %acc_value: f32, %acc_index: i32) {
      %gt = arith.cmpf ogt, %value, %acc_value : f32
      %new_value = arith.select %gt, %value, %acc_value : f32
      %new_index = arith.select %gt, %index, %acc_index : i32
      linalg.yield %new_value, %new_index : f32, i32
    }
  return
}

// CHECK-LABEL: func.func @reduce_argmax_pair
// CHECK-NOT: linalg.reduce
// CHECK: scf.for
// CHECK: memref.load %arg0
// CHECK: memref.load %arg2[]
// CHECK: arith.cmpf
// CHECK: arith.select
// CHECK: memref.store {{.*}}, %arg2[]
// CHECK: memref.store {{.*}}, %arg3[]

// -----
// CASE: linalg-fill-rank0-native-store.mlir
func.func @fill_rank0(%value: f32, %out: memref<f32>) {
  linalg.fill ins(%value : f32) outs(%out : memref<f32>)
  return
}

// CHECK-LABEL: func.func @fill_rank0
// CHECK-NOT: linalg.fill
// CHECK: memref.store %arg0, %arg1[] : memref<f32>

// -----
// CASE: linalg-index-generic-to-scf-fallback.mlir
func.func @index_init(%out: memref<16xi32>) {
  linalg.generic {indexing_maps = [affine_map<(i)->(i)>], iterator_types = ["parallel"]}
      outs(%out : memref<16xi32>) {
    ^bb0(%o: i32):
      %idx = linalg.index 0 : index
      %v = arith.index_cast %idx : index to i32
      linalg.yield %v : i32
  }
  return
}

// CHECK-LABEL: func.func @index_init
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.index
// CHECK: scf.for
// CHECK: arith.index_cast
// CHECK: memref.store

// -----
// CASE: linalg-generic-cast-to-scf-fallback-extf-f16-f32.mlir
func.func @cast_extf(%arg0: memref<16xf16>, %arg1: memref<16xf32>) {
  linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
      ins(%arg0 : memref<16xf16>)
      outs(%arg1 : memref<16xf32>) {
    ^bb0(%in: f16, %out: f32):
      %v = arith.extf %in : f16 to f32
      linalg.yield %v : f32
  }
  return
}

// CHECK-LABEL: func.func @cast_extf
// CHECK-NOT: linalg.generic
// CHECK: vir.set_vl
// CHECK: vir.extf
// CHECK: vir.store

// -----
// CASE: linalg-generic-cast-to-scf-fallback-truncf-f32-f16.mlir
func.func @cast_truncf(%arg0: memref<16xf32>, %arg1: memref<16xf16>) {
  linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
      ins(%arg0 : memref<16xf32>)
      outs(%arg1 : memref<16xf16>) {
    ^bb0(%in: f32, %out: f16):
      %v = arith.truncf %in : f32 to f16
      linalg.yield %v : f16
  }
  return
}

// CHECK-LABEL: func.func @cast_truncf
// CHECK-NOT: linalg.generic
// CHECK: vir.set_vl
// CHECK: vir.truncf
// CHECK: vir.store

// -----
// CASE: non-extf/truncf casts still use scalar fallback.
func.func @cast_sitofp_fallback(%arg0: memref<16xi32>, %arg1: memref<16xf32>) {
  linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
      ins(%arg0 : memref<16xi32>)
      outs(%arg1 : memref<16xf32>) {
    ^bb0(%in: i32, %out: f32):
      %v = arith.sitofp %in : i32 to f32
      linalg.yield %v : f32
  }
  return
}

// CHECK-LABEL: func.func @cast_sitofp_fallback
// CHECK-NOT: linalg.generic
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.sitofp
// CHECK: memref.store
