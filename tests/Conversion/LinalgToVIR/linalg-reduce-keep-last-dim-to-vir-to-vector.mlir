// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" -cse --convert-vector-to-scf --lower-affine --expand-strided-metadata --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-runner -O0 -e main -entry-point-result=i32 -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

// RUN: buddy-opt %s -lower-linalg-to-vir | FileCheck %s --check-prefix=VIR

// VIR-LABEL: func.func @sum_d1
// VIR: vir.set_vl
// VIR: scf.for
// VIR: vir.load %{{.*}} : memref<2x3x2x5xf32> -> !vir.vec<?xf32>
// VIR: arith.addf %{{.*}}, %{{.*}} : !vir.vec<?xf32>
// VIR: vir.store
// VIR-LABEL: func.func @max_d1
// VIR: arith.maximumf %{{.*}}, %{{.*}} : !vir.vec<?xf32>
// VIR-LABEL: func.func @sum_d02
// VIR-NOT: vir.reduce
// VIR-NOT: linalg.reduce
// VIR-LABEL: func.func @sum_dyn_d0
// VIR: vir.load %{{.*}} : memref<?x?xf32> -> !vir.vec<?xf32>
// VIR-NOT: vir.reduce
// VIR-LABEL: func.func @sum_2d_dyn
// VIR: vir.set_vl
// VIR-NOT: linalg.reduce
// VIR-LABEL: func.func @sum_2d_dyn_m
// VIR: vir.set_vl
// VIR-NOT: linalg.reduce
// VIR-LABEL: func.func @sum_2d_dyn_n
// VIR: vir.set_vl
// VIR-NOT: linalg.reduce
// VIR-LABEL: func.func @sum_3d_dyn_m
// VIR: vir.set_vl
// VIR-NOT: linalg.reduce
// VIR-LABEL: func.func @sum_3d_dyn_n
// VIR: vir.set_vl
// VIR-NOT: linalg.reduce
// VIR-LABEL: func.func @sum_3d_dyn_k
// VIR: vir.set_vl
// VIR-NOT: linalg.reduce
// VIR-LABEL: func.func @main

// linalg.reduce that keeps the last dim: vectorized along it, the slices
// along the reduced dims combined elementwise (no vir.reduce).

memref.global "private" constant @x : memref<2x3x2x5xf32> = dense<[[[[-5.0, -3.0, -1.0, 1.0, 3.0], [-2.0, 0.0, 2.0, 4.0, -5.0]], [[0.0, 2.0, 4.0, -5.0, -3.0], [3.0, 5.0, -4.0, -2.0, 0.0]], [[5.0, -4.0, -2.0, 0.0, 2.0], [-3.0, -1.0, 1.0, 3.0, 5.0]]], [[[2.0, 4.0, -5.0, -3.0, -1.0], [5.0, -4.0, -2.0, 0.0, 2.0]], [[-4.0, -2.0, 0.0, 2.0, 4.0], [-1.0, 1.0, 3.0, 5.0, -4.0]], [[1.0, 3.0, 5.0, -4.0, -2.0], [4.0, -5.0, -3.0, -1.0, 1.0]]]]>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @sum_d1(%x: memref<2x3x2x5xf32>, %o: memref<2x2x5xf32>) {
  linalg.reduce ins(%x : memref<2x3x2x5xf32>) outs(%o : memref<2x2x5xf32>)
    dimensions = [1]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return
}

func.func @max_d1(%x: memref<2x3x2x5xf32>, %o: memref<2x2x5xf32>) {
  linalg.reduce ins(%x : memref<2x3x2x5xf32>) outs(%o : memref<2x2x5xf32>)
    dimensions = [1]
    (%in: f32, %acc: f32) {
      %m = arith.maximumf %in, %acc : f32
      linalg.yield %m : f32
    }
  return
}

func.func @sum_d02(%x: memref<2x3x2x5xf32>, %o: memref<3x5xf32>) {
  linalg.reduce ins(%x : memref<2x3x2x5xf32>) outs(%o : memref<3x5xf32>)
    dimensions = [0, 2]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return
}

// Dynamic shapes, dimensions = [0] (the scalar-loop lowering of a rank-2
// reduction over dim 0 needs static shapes).
func.func @sum_dyn_d0(%x: memref<?x?xf32>, %o: memref<?xf32>) {
  linalg.reduce ins(%x : memref<?x?xf32>) outs(%o : memref<?xf32>)
    dimensions = [0]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return
}

// The other shapes over dimensions = [0] with a dynamic dim, lowered as
// @sum_dyn_d0.
func.func @sum_2d_dyn(%x: memref<?x?xf32>, %o: memref<?xf32>) {
  linalg.reduce ins(%x : memref<?x?xf32>) outs(%o : memref<?xf32>)
    dimensions = [0]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return
}

func.func @sum_2d_dyn_m(%x: memref<?x8xf32>, %o: memref<8xf32>) {
  linalg.reduce ins(%x : memref<?x8xf32>) outs(%o : memref<8xf32>)
    dimensions = [0]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return
}

func.func @sum_2d_dyn_n(%x: memref<4x?xf32>, %o: memref<?xf32>) {
  linalg.reduce ins(%x : memref<4x?xf32>) outs(%o : memref<?xf32>)
    dimensions = [0]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return
}

func.func @sum_3d_dyn_m(%x: memref<?x4x8xf32>, %o: memref<4x8xf32>) {
  linalg.reduce ins(%x : memref<?x4x8xf32>) outs(%o : memref<4x8xf32>)
    dimensions = [0]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return
}

func.func @sum_3d_dyn_n(%x: memref<2x?x8xf32>, %o: memref<?x8xf32>) {
  linalg.reduce ins(%x : memref<2x?x8xf32>) outs(%o : memref<?x8xf32>)
    dimensions = [0]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return
}

func.func @sum_3d_dyn_k(%x: memref<2x4x?xf32>, %o: memref<4x?xf32>) {
  linalg.reduce ins(%x : memref<2x4x?xf32>) outs(%o : memref<4x?xf32>)
    dimensions = [0]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return
}

func.func @main() -> i32 {
  %x = memref.get_global @x : memref<2x3x2x5xf32>
  %one = arith.constant 1.0 : f32
  %m2 = arith.constant -2.0 : f32
  %zero = arith.constant 0.0 : f32
  %o1 = memref.alloc() : memref<2x2x5xf32>
  linalg.fill ins(%one : f32) outs(%o1 : memref<2x2x5xf32>)
  call @sum_d1(%x, %o1) : (memref<2x3x2x5xf32>, memref<2x2x5xf32>) -> ()
  %p1 = memref.cast %o1 : memref<2x2x5xf32> to memref<*xf32>
  call @printMemrefF32(%p1) : (memref<*xf32>) -> ()
  %o2 = memref.alloc() : memref<2x2x5xf32>
  linalg.fill ins(%m2 : f32) outs(%o2 : memref<2x2x5xf32>)
  call @max_d1(%x, %o2) : (memref<2x3x2x5xf32>, memref<2x2x5xf32>) -> ()
  %p2 = memref.cast %o2 : memref<2x2x5xf32> to memref<*xf32>
  call @printMemrefF32(%p2) : (memref<*xf32>) -> ()
  %o3 = memref.alloc() : memref<3x5xf32>
  linalg.fill ins(%zero : f32) outs(%o3 : memref<3x5xf32>)
  call @sum_d02(%x, %o3) : (memref<2x3x2x5xf32>, memref<3x5xf32>) -> ()
  %p3 = memref.cast %o3 : memref<3x5xf32> to memref<*xf32>
  call @printMemrefF32(%p3) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 3 offset = 0 sizes = \[2, 2, 5\]}}
  // CHECK-NEXT{LITERAL}: [[[1, -4, 2, -3, 3],
  // CHECK-NEXT{LITERAL}: [-1, 5, 0, 6, 1]],
  // CHECK-NEXT{LITERAL}: [[0, 6, 1, -4, 2],
  // CHECK-NEXT{LITERAL}: [9, -7, -1, 5, 0]]]
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 3 offset = 0 sizes = \[2, 2, 5\]}}
  // CHECK-NEXT{LITERAL}: [[[5, 2, 4, 1, 3],
  // CHECK-NEXT{LITERAL}: [3, 5, 2, 4, 5]],
  // CHECK-NEXT{LITERAL}: [[2, 4, 5, 2, 4],
  // CHECK-NEXT{LITERAL}: [5, 1, 3, 5, 2]]]
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[3, 5\]}}
  // CHECK-NEXT{LITERAL}: [[0, -3, -6, 2, -1],
  // CHECK-NEXT{LITERAL}: [-2, 6, 3, 0, -3],
  // CHECK-NEXT{LITERAL}: [7, -7, 1, -2, 6]]
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 1 offset = 0 sizes = \[10\]}}
  // CHECK-NEXT{LITERAL}: [-1, 0, 1, -9, 3, 6, -4, -3, 9, -1]
  %x2 = memref.collapse_shape %x [[0, 1], [2, 3]]
    : memref<2x3x2x5xf32> into memref<6x10xf32>
  %xd = memref.cast %x2 : memref<6x10xf32> to memref<?x?xf32>
  %o4 = memref.alloc() : memref<10xf32>
  linalg.fill ins(%zero : f32) outs(%o4 : memref<10xf32>)
  %od = memref.cast %o4 : memref<10xf32> to memref<?xf32>
  call @sum_dyn_d0(%xd, %od) : (memref<?x?xf32>, memref<?xf32>) -> ()
  %p4 = memref.cast %o4 : memref<10xf32> to memref<*xf32>
  call @printMemrefF32(%p4) : (memref<*xf32>) -> ()
  memref.dealloc %o4 : memref<10xf32>
  memref.dealloc %o1 : memref<2x2x5xf32>
  memref.dealloc %o2 : memref<2x2x5xf32>
  memref.dealloc %o3 : memref<3x5xf32>
  %ret = arith.constant 0 : i32
  return %ret : i32
}
