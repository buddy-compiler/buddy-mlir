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

func.func @main() -> i32 {
  // vector.insert_strided_slice can insert a vector into an area of bigger
  // vector with offsets and strides.

  // Unlike vector.insert, which could only insert "element"/"sub-vector" into a vector,
  // vector.insert_strided_slice let you insert data in small vector almost
  // freely into any where of the base vector, using a "offset-strides" style.

  %base = arith.constant dense<[[0, 1, 2, 3], [10, 11, 12, 13],
                                [20, 21, 22, 23], [30, 31, 32, 33]]> : vector<4x4xi32>

  %val = arith.constant dense<[[100, 101], [110, 111]]> : vector<2x2xi32>


  // With offsets = [0, 0], strides = [1, 1], the insertion does:
  // x x o o
  // x x o o
  // o o o o
  // o o o o
  %w0 = vector.insert_strided_slice %val, %base { offsets = [0, 0], strides = [1, 1] }
    : vector<2x2xi32> into vector<4x4xi32>
  // CHECK:    ( ( 100, 101, 2, 3 ),
  // CHECK-SAME: ( 110, 111, 12, 13 ),
  // CHECK-SAME: ( 20, 21, 22, 23 ),
  // CHECK-SAME: ( 30, 31, 32, 33 ) )
  vector.print %w0 : vector<4x4xi32>


  // we can add offsets to both dim 0 and dim 1:
  // when strides = [1, 1], offset =
  // [1, 0]:  o o o o    | [0, 1]:  o x x o   | [1, 1]: o o o o
  //          x x o o    |          o x x o   |         o x x o
  //          x x o o    |          o o o o   |         o x x o
  //          o o o o    |          o o o o   |         o o o o
  %w1_0 = vector.insert_strided_slice %val, %base { offsets = [1, 0], strides = [1, 1] }
    : vector<2x2xi32> into vector<4x4xi32>
  %w1_1 = vector.insert_strided_slice %val, %base { offsets = [0, 1], strides = [1, 1] }
    : vector<2x2xi32> into vector<4x4xi32>
  %w1_2 = vector.insert_strided_slice %val, %base { offsets = [1, 1], strides = [1, 1] }
    : vector<2x2xi32> into vector<4x4xi32>
  // CHECK:      ( ( 0, 1, 2, 3 ),
  // CHECK-SAME: ( 100, 101, 12, 13 ),
  // CHECK-SAME: ( 110, 111, 22, 23 ),
  // CHECK-SAME: ( 30, 31, 32, 33 ) )
  vector.print %w1_0 : vector<4x4xi32>
  // CHECK:     ( ( 0, 100, 101, 3 ),
  // CHECK-SMAE: ( 10, 110, 111, 13 ),
  // CHECK-SAME: ( 20, 21, 22, 23 ),
  // CHECK-SAME: ( 30, 31, 32, 33 ) )
  vector.print %w1_1 : vector<4x4xi32>
  // CHECK:     ( ( 0, 1, 2, 3 ),
  // CHECK-SAME: ( 10, 100, 101, 13 ),
  // CHECK-SMAE: ( 20, 110, 111, 23 ),
  // CHECK-SMAE: ( 30, 31, 32, 33 ) )
  vector.print %w1_2 : vector<4x4xi32>


  // NEED-IMPL: strides are used to specify how sub-vector are inserted with steps
  // Currently strides only allow 1s, so the example above could not work yet.
  // when offset = [0, 0], strides =
  // [2, 1]:  x x o o    | [1, 2]:  x o x o   | [2, 2]: x o x o
  //          o o o o    |          x o x o   |         o o o o
  //          x x o o    |          o o o o   |         x o x o
  //          o o o o    |          o o o o   |         o o o o

  // %w2_0 = vector.insert_strided_slice %val, %base { offsets = [0, 0], strides = [2, 1] }
  //   : vector<2x2xi32> into vector<4x4xi32>
  // %w2_1 = vector.insert_strided_slice %val, %base { offsets = [0, 0], strides = [1, 2] }
  //   : vector<2x2xi32> into vector<4x4xi32>
  // %w2_2 = vector.insert_strided_slice %val, %base { offsets = [0, 0], strides = [2, 2] }
  //   : vector<2x2xi32> into vector<4x4xi32>

  // vector.print %w2_0 : vector<4x4xi32>
  // vector.print %w2_1 : vector<4x4xi32>
  // vector.print %w2_2 : vector<4x4xi32>


  // vector.insert_strided_slice with any rank can be defined recursively:

  // vector.insert_strided_slice %v, %b { offsets = [o0, ...], strides = [s0, ...] }
  // <==>
  // if rank(%v) < rank(%b):
  //    vector.insert_strided_slice %v, %b[o0] { offsets = [...], strides = [...] }
  //
  // if rank(%v) == rank(%b):
  //    for (i = o0; i < dim(%v); i += s0)
  //      vector.insert_strided_slice %v[i], %b[i] { offsets = [...], strides = [...] }

  %big_base = arith.constant dense<[
    [[0, 10, 20, 30], [100, 110, 120, 130], [200, 210, 220, 230], [300, 310, 320, 330]],
    [[1, 11, 21, 31], [101, 111, 121, 131], [201, 211, 221, 231], [301, 311, 321, 331]],
    [[2, 12, 22, 32], [102, 112, 122, 132], [202, 212, 222, 232], [302, 312, 322, 332]],
    [[3, 13, 23, 33], [103, 113, 123, 133], [203, 213, 223, 233], [303, 313, 323, 333]]]>
  : vector<4x4x4xi32>

  %big_value = arith.constant dense<[[1000, 1001, 1002],
                                     [1010, 1011, 1012],
                                     [1020, 1021, 1022]]> : vector<3x3xi32>

  %w3 = vector.insert_strided_slice %big_value, %big_base {offsets = [1, 0, 0], strides = [1, 1]}
    : vector<3x3xi32> into vector<4x4x4xi32>
  // CHECK:  ( ( ( 0, 10, 20, 30 ), ( 100, 110, 120, 130 ), ( 200, 210, 220, 230 ), ( 300, 310, 320, 330 ) ),
  // CHECK-SMAE: ( ( 1000, 1001, 1002, 31 ), ( 1010, 1011, 1012, 131 ), ( 1020, 1021, 1022, 231 ), ( 301, 311, 321, 331 ) ),
  // CHECK-SMAE: ( ( 2, 12, 22, 32 ), ( 102, 112, 122, 132 ), ( 202, 212, 222, 232 ), ( 302, 312, 322, 332 ) ),
  // CHECK-SMAE: ( ( 3, 13, 23, 33 ), ( 103, 113, 123, 133 ), ( 203, 213, 223, 233 ), ( 303, 313, 323, 333 ) ) )
  vector.print %w3 : vector<4x4x4xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
