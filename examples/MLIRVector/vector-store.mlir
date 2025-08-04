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
  // vector.store can store n-D vector into m-D memref

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

  // store normal usage
  %value0 = arith.constant dense<[100, 101, 102]> : vector<3xi32>
  vector.store %value0, %base0[%c0] : memref<8xi32> , vector<3xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [8] strides = [1] data =
  // CHECK-NEXT: [100,  101,  102,  3,  4,  5,  6,  7]
  func.call @printMemrefI32(%gv0_for_print) : (memref<*xi32>) -> ()


  // store with multi-dimension memref
  //    case 1: inside most-inner dimension
  %value1 = arith.constant dense<[200, 201, 202, 203]> : vector<4xi32>
  vector.store %value1, %base1[%c0, %c0] : memref<4x4xi32>, vector<4xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [200,   201,   202,   203],
  // CHECK-NEXT: [4,   5,   6,   7],
  // CHECK-NEXT: [8,   9,   10,   11],
  // CHECK-NEXT: [12,   13,   14,   15]
  // CHECK-SAME: ]
  func.call @printMemrefI32(%gv1_for_print) : (memref<*xi32>) -> ()


  // store with multi-dimension memref
  //    case 2: cross the most-inner dimension
  // In this case, it will behave like the memref is flat
  %value2 = arith.constant dense<[300, 301, 302, 303, 304, 305, 306, 307]> : vector<8xi32>
  vector.store %value2, %base1[%c0, %c0] : memref<4x4xi32>, vector<8xi32>
  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [300,   301,   302,   303],
  // CHECK-NEXT: [304,   305,   306,   307],
  // CHECK-NEXT:  [8,   9,   10,   11],
  // CHECK-NEXT: [12,   13,   14,   15]
  // CHECK-SAME: ]
  func.call @printMemrefI32(%gv1_for_print) : (memref<*xi32>) -> ()


  // store into memref of vector
  %base4 = memref.alloc() : memref<2xvector<4xi32>>

  %value3_0 = arith.constant dense<[300, 301, 302, 303]> : vector<4xi32>
  vector.store %value3_0, %base4[%c0] : memref<2xvector<4xi32>>, vector<4xi32>

  // this one fails
  // %value3_1 = arith.constant dense<[[310, 311], [312, 312]]> : vector<2x2xi32>
  // vector.store %value3_1, %base4[%c1] : memref<2xvector<4xi32>>, vector<2x2xi32>

  // TODO: Printed results with errors, currently skipping value test.
  // CHECK: {{.*}}

  // Use a for loop to print the whole memref
  // equal to "for i = 0 to 2 { print %base4[i] }"
  scf.for %i = %c0 to %c2 step %c1 {
    %v = memref.load %base4[%i] : memref<2xvector<4xi32>>
    vector.print %v : vector<4xi32>
  }


  // store an n-D vector into memref
  // TODO: figure out why it failed. The document says it SHOULD work.

  // %value4 = arith.constant dense<[[400, 401], [402, 403]]> : vector<2x2xi32>
  // vector.store %value4, %base1[%c2, %c0] : memref<4x4xi32>, vector<2x2xi32>
  // vector.store %value4, %base3[%c2] : memref<8xi32>, vector<2x2xi32>
  // func.call @printMemrefI32(%gv3_for_print) : (memref<*xi32>) -> ()


  // store with memref with custom layout
  // TODO: find out how to create a memref with arbitrarily affine map layout
  // "5" is reserved for this example

  //============================================================================
  // Tips: because keep using the same memory region for all examples will make
  // the changes of memref look very messed up, we change to another clean memref
  // as our "base ptr" below. (%gv2)
  //============================================================================

  // store with dynamic memref
  //    case 1: in-bound
  %base6 = memref.cast %base2 : memref<4x4xi32> to memref<?x?xi32>
  %mask6 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  %value6 = arith.constant dense<[600, 601, 602, 603, 604, 605, 606, 607]> : vector<8xi32>

  vector.store %value6, %base6[%c1, %c1] : memref<?x?xi32>, vector<8xi32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   1,   2,   3],
  // CHECK-NEXT: [4,   600,   601,   602],
  // CHECK-NEXT: [603,   604,   605,   606],
  // CHECK-NEXT: [607,   13,   14,   15]
  // CHECK-SAME: ]

  func.call @printMemrefI32(%gv2_for_print) : (memref<*xi32>) -> ()


  // store with dynamic memref
  //    case 2: out of bound
  // The document says:
  //    Representation-wise, the ‘vector.store’ operation permits out-of-bounds writes.
  //    Support and implementation of out-of-bounds vector stores are target-specific.
  //    No assumptions should be made on the memory written out of bounds.
  //    Not all targets may support out-of-bounds vector stores.
  %base7 = memref.cast %base2 : memref<4x4xi32> to memref<?x?xi32>
  %value7 = arith.constant dense<[700, 701, 702, 703, 704, 705, 706, 707]> : vector<8xi32>

  vector.store %value7, %base7[%c3, %c1] : memref<?x?xi32>, vector<8xi32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME: [0,   1,   2,   3],
  // CHECK-NEXT: [4,   600,   601,   602],
  // CHECK-NEXT: [603,   604,   605,   606],
  // CHECK-NEXT: [607,   700,   701,   702]
  // CHECK-SAME: ]
  // if you are lucky, you will see that gv3 is changed by out-of-bound writing
  func.call @printMemrefI32(%gv2_for_print) : (memref<*xi32>) -> ()
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [8] strides = [1] data =
  // CHECK-NEXT: [703,  704,  705,  706,  707,  5,  6,  7]
  func.call @printMemrefI32(%gv3_for_print) : (memref<*xi32>) -> ()


  // store with unranked memref is not allowed
  %base8 = memref.cast %base2 : memref<4x4xi32> to memref<*xi32>
  %mask8 = arith.constant dense<[1, 0, 0, 1]> : vector<4xi1>
  %value8 = arith.constant dense<[800, 801, 802, 803]> : vector<4xi32>

  // vector.maskedstore %base8[%c0, %c0], %mask8, %value8
  //   : memref<*xi32>, vector<4xi1>, vector<4xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
