// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -split-input-file -verify-diagnostics \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner  -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @gv0 : memref<2x4xi32> = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]>
memref.global "private" @gv1 : memref<2x3xi32> = dense<[[1, 2, 3], [4, 5, 6]]>
memref.global "private" @gv2 : memref<2xvector<3xi32>>

func.func @main() -> i32 {
  // vector.type_cast casts:
  //      memref<a... x T> ==> memref<vector<a... x T>>
  //      memref<a... x vector<b... x T>> => memref<vector<a... x b... x T>>
  // It makes a region in memory to be viewed as **one** vector that we can do
  // all the vector operations on it.


  // preparation for examples
  %mem0 = memref.get_global @gv0 : memref<2x4xi32>
  %mem1 = memref.get_global @gv1 : memref<2x3xi32>
  %mem2 = memref.get_global @gv2 : memref<2xvector<3xi32>>

  // init @gv2, setting it to [[1, 2, 3], [4, 5, 6]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %t0 = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %t1 = arith.constant dense<[4, 5, 6]> : vector<3xi32>

  memref.store %t0, %mem2[%c0] : memref<2xvector<3xi32>>
  memref.store %t1, %mem2[%c1] : memref<2xvector<3xi32>>


  // Normal usage
  %m0 = vector.type_cast %mem0 : memref<2x4xi32> to memref<vector<2x4xi32>>

  %v0 = memref.load %m0[] : memref<vector<2x4xi32>>
  // CHECK: ( ( 1, 2, 3, 4 ), ( 5, 6, 7, 8 ) )
  vector.print %v0 : vector<2x4xi32>


  // LLVM backend only supports 1-D vectors and requires them to align to A byte
  // in an array, where A = length of vector rounded up to the power of 2.
  // In current lowering path, MLIR's n-D vector<a... x b x T> is translated to
  // LLVM's [a... x vector<b x T>].

  // So if we want to tell the compiler that "this memory region should be
  // viewed as a vector<a... x b x T>", we should manually align the vectors,
  // which unfortunately will break the abstraction that memref provided and
  // introduce too many implementation details.

  // This is buggy, so applying vector.type_cast to memref<a... x b x T>
  // when b is not a power of 2 is not recommended yet, until we have a way to
  // express platform-specific DataLayout (which contains align requirement) in MLIR.

  // If it hasn't be fixed, you will see "( ( 1, 2, 3 ), ( 5, 6, 0 ) )" instead
  // of "( ( 1, 2, 3 ), ( 4, 5, 6 ) )" with LLVM backend.
  // %m1 = vector.type_cast %mem1 : memref<2x3xi32> to memref<vector<2x3xi32>>
  // %v1 = memref.load %m1[] : memref<vector<2x3xi32>>
  // vector.print %v1 : vector<2x3xi32>


  // However, applying vector.type_cast on memref<a... x vector<b... x T>> will
  // not produce any buggy behaviors because align requirement is satisfied when
  // we init the memref and store things into it.
  %m2 = vector.type_cast %mem2 : memref<2xvector<3xi32>> to memref<vector<2x3xi32>>

  %v2 = memref.load %m2[] : memref<vector<2x3xi32>>
  //CHECK: ( ( 1, 2, 3 ), ( 4, 5, 6 ) )
  vector.print %v2 : vector<2x3xi32>


  %ret = arith.constant 0 : i32
  return %ret : i32
}
