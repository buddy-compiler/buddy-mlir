// RUN: buddy-opt %s \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %sum_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
  %sum = affine.for %i = 0 to 3 iter_args(%sum_iter = %sum_0) -> (vector<4xf32>) {
    %load_vec1 = vector.load %mem[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
    %load_vec2 = vector.load %mem[%i, %c0] : memref<4x4xf32>, vector<4xf32>
    %sum_next = vector.fma %load_vec1, %load_vec2, %sum_iter : vector<4xf32>
    affine.yield %sum_next : vector<4xf32>
  }
  // CHECK: ( 0, 33, 72, 117 )
  vector.print %sum : vector<4xf32>
  %ret = arith.constant 0 : i32
  return %ret : i32
}
