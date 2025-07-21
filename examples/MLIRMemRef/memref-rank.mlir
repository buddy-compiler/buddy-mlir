// RUN: buddy-opt %s \
// RUN:     -lower-affine -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-vector-to-llvm -convert-func-to-llvm  \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
func.func @main() {
  %mem0 = memref.alloc() : memref<4xf32>
  %mem1 = memref.alloc() : memref<4x4xf32>
  %mem2 = memref.cast %mem1 : memref<4x4xf32> to memref<*xf32>
  %i0 = memref.rank %mem0 : memref<4xf32>
  %i1 = memref.rank %mem1 : memref<4x4xf32>
  %i2 = memref.rank %mem2 : memref<*xf32>
  // CHECK: 1
  vector.print %i0 : index
  // CHECK: 2
  vector.print %i1 : index
  // CHECK: 2
  vector.print %i2 : index
  memref.dealloc %mem0 : memref<4xf32>
  memref.dealloc %mem1 : memref<4x4xf32>
  return
}
