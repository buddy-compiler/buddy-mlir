// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1, d0)>

func.func @main() -> (i32) {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %mem = memref.alloc() : memref<4x4xf32>
  %mem_cast = memref.cast %mem : memref<4x4xf32> to memref<*xf32>
  %v0 = arith.constant dense<[[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.]]> : vector<4x4xf32>
  %v1 = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : vector<2x3xf32>
  vector.transfer_write %v0, %mem[%cst0, %cst0] { } : vector<4x4xf32>, memref<4x4xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   0,   0,   0],
  // CHECK-NEXT: [0,   0,   0,   0],
  // CHECK-NEXT: [0,   0,   0,   0],
  // CHECK-NEXT: [0,   0,   0,   0]
  // CHECK-SAME: ]

  call @printMemrefF32(%mem_cast) : (memref<*xf32>) -> ()
  vector.transfer_write %v1, %mem[%cst1, %cst1] { } : vector<2x3xf32>, memref<4x4xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   0,   0,   0],
  // CHECK-NEXT: [0,   1,   2,   3],
  // CHECK-NEXT: [0,   4,   5,   6],
  // CHECK-NEXT: [0,   0,   0,   0]
  // CHECK-SAME: ]

  call @printMemrefF32(%mem_cast) : (memref<*xf32>) -> ()
  vector.transfer_write %v1, %mem[%cst1, %cst1] { permutation_map = #map0 } : vector<2x3xf32>, memref<4x4xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   0,   0,   0],
  // CHECK-NEXT: [0,   1,   4,   3],
  // CHECK-NEXT: [0,   2,   5,   6],
  // CHECK-NEXT: [0,   3,   6,   0]
  // CHECK-SAME: ]
  call @printMemrefF32(%mem_cast) : (memref<*xf32>) -> ()
  memref.dealloc %mem : memref<4x4xf32>
  return %c0 : i32
}

func.func private @printMemrefF32(memref<*xf32>)
