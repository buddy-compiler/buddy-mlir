// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -split-input-file -verify-diagnostics -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @gv0 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>

memref.global "private" @gv1 : memref<4x4xi32> = dense<[[0, 1, 2, 3],
                                                        [4, 5, 6, 7],
                                                        [8, 9, 10, 11],
                                                        [12, 13, 14, 15]]>

memref.global "private" @gv2 : memref<4x4xi32> = dense<[[0, 1, 2, 3],
                                                        [4, 5, 6, 7],
                                                        [8, 9, 10, 11],
                                                        [12, 13, 14, 15]]>

memref.global "private" @gv3 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @main() -> i32 {
  // maskedstore is a store with a mask, supporting store a 1-D vector
  // into an n-D memref with the mask.

  // preparation for examples
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %base0 = memref.get_global @gv0 : memref<8xi32>
  %base1 = memref.get_global @gv1 : memref<4x4xi32>
  %base2 = memref.get_global @gv2 : memref<4x4xi32>
  %base3 = memref.get_global @gv3 : memref<8xi32>

  %gv0_for_print = memref.cast %base0 : memref<8xi32> to memref<*xi32>
  %gv1_for_print = memref.cast %base1 : memref<4x4xi32> to memref<*xi32>
  %gv2_for_print = memref.cast %base2 : memref<4x4xi32> to memref<*xi32>
  %gv3_for_print = memref.cast %base3 : memref<8xi32> to memref<*xi32>

  // maskedstore normal usage
  %mask0 = arith.constant dense<[1, 0, 1]> : vector<3xi1>
  %value0 = arith.constant dense<[100, 101, 102]> : vector<3xi32>

  vector.maskedstore %base0[%c0], %mask0, %value0
    : memref<8xi32> , vector<3xi1>,vector<3xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [8] strides = [1] data =
  // CHECK-NEXT: [100,  1,  102,  3,  4,  5,  6,  7]
  func.call @printMemrefI32(%gv0_for_print) : (memref<*xi32>) -> ()


  // maskedstore with multi-dimension memref
  //    case 1: inside most-inner dimension
  %mask1 = arith.constant dense<[1, 0, 0, 1]> : vector<4xi1>
  %value1 = arith.constant dense<[200, 201, 202, 203]> : vector<4xi32>

  vector.maskedstore %base1[%c0, %c0], %mask1, %value1
    : memref<4x4xi32>, vector<4xi1>, vector<4xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [200,   1,   2,   203],
  // CHECK-NEXT: [4,   5,   6,   7],
  // CHECK-NEXT: [8,   9,   10,   11],
  // CHECK-NEXT: [12,   13,   14,   15]
  // CHECK-SAME: ]
  func.call @printMemrefI32(%gv1_for_print) : (memref<*xi32>) -> ()


  // maskedstore with multi-dimension memref
  //    case 2: cross the most-inner dimension
  // In this case, it will behave like the memref is flat
  %mask2 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %value2 = arith.constant dense<[300, 301, 302, 303, 304, 305, 306, 307]> : vector<8xi32>

  vector.maskedstore %base1[%c0, %c0], %mask2, %value2
    : memref<4x4xi32> , vector<8xi1>,vector<8xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [300,   1,   302,   303],
  // CHECK-NEXT: [304,   305,   6,   7],
  // CHECK-NEXT: [8,   9,   10,   11],
  // CHECK-NEXT: [12,   13,   14,   15]
  // CHECK-SAME:]
  func.call @printMemrefI32(%gv1_for_print) : (memref<*xi32>) -> ()


  // maskedstore with memref with custom layout
  // TODO: find out how to create a memref with arbitrarily affine map layout
  // "3" is reserved for this example

  //============================================================================
  // Tips: because keep using the same memory region for all examples will make the
  // changes of memref look very messed up, we change to another clean memref
  // as our "base ptr" below. (%gv2)
  //============================================================================

  // maskedstore with dynamic memref
  //    case 1: in-bound
  %base4 = memref.cast %base2 : memref<4x4xi32> to memref<?x?xi32>
  %mask4 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %value4 = arith.constant dense<[400, 401, 402, 403, 404, 405, 406, 407]> : vector<8xi32>

  vector.maskedstore %base4[%c1, %c1], %mask4, %value4
    : memref<?x?xi32>, vector<8xi1>, vector<8xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   1,   2,   3],
  // CHECK-NEXT: [4,   400,   6,   402],
  // CHECK-NEXT: [403,   404,   405,   11],
  // CHECK-NEXT: [12,   13,   14,   15]
  // CHECK-SAME: ]
  func.call @printMemrefI32(%gv2_for_print) : (memref<*xi32>) -> ()


  // maskedstore with dynamic memref
  //    case 2: what will happen if we store into somewhere out of bounds?
  %base5 = memref.cast %base2 : memref<4x4xi32> to memref<?x?xi32>
  %mask5 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %value5 = arith.constant dense<[500, 501, 502, 503, 504, 505, 506, 507]> : vector<8xi32>

  vector.maskedstore %base5[%c3, %c1], %mask5, %value5
    : memref<?x?xi32>, vector<8xi1>, vector<8xi32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   1,   2,   3],
  // CHECK-NEXT: [4,   400,   6,   402],
  // CHECK-NEXT: [403,   404,   405,   11],
  // CHECK-NEXT: [12,   500,   14,   502]
  // CHECK-SMAE: ]

  // the @gv2 looks good
  func.call @printMemrefI32(%gv2_for_print) : (memref<*xi32>) -> ()

  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [8] strides = [1] data =
  // CHECK-NEXT: [503,  504,  505,  3,  4,  5,  6,  7]

  // oops, we write the rest part to @gv3
  func.call @printMemrefI32(%gv3_for_print) : (memref<*xi32>) -> ()


  // maskedstore with unranked memref is not allowed
  // %base6 = memref.cast %base2 : memref<4x4xi32> to memref<*xi32>
  // %mask6 = arith.constant dense<[1, 0, 0, 1]> : vector<4xi1>
  // %value6 = arith.constant dense<[600, 601, 602, 603]> : vector<4xi32>

  // vector.maskedstore %base6[%c0, %c0], %mask6, %value6
  //   : memref<*xi32>, vector<4xi1>, vector<4xi32>

  // Unlike store, maskedstore with n-D vector is not allowed
  // %mask7 = arith.constant dense<[[1, 0, 0, 1], [0, 1, 1, 0]]> : vector<2x4xi1>
  // %value7 = arith.constant dense<[[700, 701, 702, 703], [704, 705, 706, 707]]> : vector<2x4xi32>

  // vector.maskedstore %base2[%c0, %c0], %mask7, %value7
  //   : memref<4x4xi32>, vector<2x4xi1>, vector<2x4xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
