// RUN: buddy-opt %s \
// RUN:     -lower-affine -finalize-memref-to-llvm \
// RUN:     -convert-arith-to-llvm -convert-vector-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mem0 = memref.alloc() : memref<2x3xf32>
  %mem1 = memref.cast %mem0 : memref<2x3xf32> to memref<?x?xf32>
  %dim0 = memref.dim %mem0, %c0 : memref<2x3xf32>
  %dim1 = memref.dim %mem0, %c1 : memref<2x3xf32>
  %dim2 = memref.dim %mem1, %c0 : memref<?x?xf32>
  %dim3 = memref.dim %mem1, %c1 : memref<?x?xf32>
  // CHECK: 2
  vector.print %dim0 : index
  // CHECK: 3
  vector.print %dim1 : index
  // CHECK: 2
  vector.print %dim2 : index
  // CHECK: 3
  vector.print %dim3 : index
  memref.dealloc %mem0 : memref<2x3xf32>
  return
}
