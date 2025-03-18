// RUN: buddy-opt %s \
// RUN:     -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm \
// RUN:		  -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
// RUN:		  -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @gv : memref<4xf32> = dense<[0. , 1. , 2. , 3. ]>
#map0 = affine_map<(d0, d1) -> (d0 + d1)>

func.func @main() {
  %mem = memref.get_global @gv : memref<4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  affine.for %idx0 = 0 to 2 {
    affine.for %idx1 = 0 to 2 {
      // Method one.
      %ele0 = affine.load %mem[%idx0 + %idx1] : memref<4xf32>
      vector.print %ele0 : f32
      // Method two.
      %idx = affine.apply #map0(%idx0, %idx1)
      %ele1 = affine.load %mem[%idx] : memref<4xf32>
      vector.print %ele1 : f32
    }
  }
  // Method three.
  %idx = arith.addi %c0, %c1 : index
  %ele = affine.load %mem[symbol(%idx)] : memref<4xf32>
  // CHECK: 1
  vector.print %ele : f32
  return
}
