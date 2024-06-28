// RUN: buddy-opt %s \
// RUN:     -convert-func-to-llvm \
// RUN:     -finalize-memrefexp-to-llvm -finalize-memref-to-llvm \
// RUN:     -convert-vector-to-llvm -reconcile-unrealized-casts  \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() {
  %1 = memref_exp.null : memref<4x4xf32>
  %2 = memref.extract_aligned_pointer_as_index %1 : memref<4x4xf32> -> index
  // CHECK: 0
  vector.print %2 : index
  return
}
