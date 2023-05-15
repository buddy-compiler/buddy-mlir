// RUN: buddy-opt %s \
// RUN:     -lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
// RUN:		  -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:		  -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @gv : memref<4xf32> = dense<[0. , 1. , 2. , 3. ]>

func.func @main() {
    %mem = memref.get_global @gv : memref<4xf32>
    %sum_0 = arith.constant 0.0 : f32
    %sum = affine.for %i = 0 to 4 iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = affine.load %mem[%i] : memref<4xf32>
        %sum_next = arith.addf %sum_iter, %t : f32
        affine.yield %sum_next : f32
    }
    // CHECK: 6
    vector.print %sum : f32
    return
}