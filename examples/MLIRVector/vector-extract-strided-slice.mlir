// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @main() -> i32 {
  // vector.extract_strided_slice can extract elements from a vector with
  // offsets and strides, then group them into a new vector.

  // Unlike vector.extract, which could only extract "element"/"sub-vector" from a vector,
  // vector.extract_strided_slice let you select element almost freely from the
  // source vector using a "offset-strides" style.

  %base = arith.constant dense<[[0, 1, 2, 3], [10, 11, 12, 13],
                                [20, 21, 22, 23], [30, 31, 32, 33]]> : vector<4x4xi32>


  // With offsets = [0, 0], strides = [1, 1], it will extract:
  // x x o o
  // x x o o
  // o o o o
  // o o o o
  %w0 = vector.extract_strided_slice %base
    { offsets = [0, 0], sizes = [2, 2], strides = [1, 1] }
    : vector<4x4xi32> to vector<2x2xi32>
  // CHECK: ( ( 0, 1 ), ( 10, 11 ) )
  vector.print %w0 : vector<2x2xi32>


  // we can add offsets to both dim 0 and dim 1:
  // when strides = [1, 1], offset =
  // [1, 0]:  o o o o    | [0, 1]:  o x x o   | [1, 1]: o o o o
  //          x x o o    |          o x x o   |         o x x o
  //          x x o o    |          o o o o   |         o x x o
  //          o o o o    |          o o o o   |         o o o o
  %w1_0 = vector.extract_strided_slice %base
    { offsets = [1, 0], sizes = [2, 2], strides = [1, 1] }
    : vector<4x4xi32> to vector<2x2xi32>
  %w1_1 = vector.extract_strided_slice %base
    { offsets = [0, 1], sizes = [2, 2], strides = [1, 1] }
    : vector<4x4xi32> to vector<2x2xi32>
  %w1_2 = vector.extract_strided_slice %base
    { offsets = [1, 1], sizes = [2, 2], strides = [1, 1] }
    : vector<4x4xi32> to vector<2x2xi32>
  // CHECK: ( ( 10, 11 ), ( 20, 21 ) )
  vector.print %w1_0 : vector<2x2xi32>
  // CHECk: ( ( 1, 2 ), ( 11, 12 ) )
  vector.print %w1_1 : vector<2x2xi32>
  // CHECk: ( ( 11, 12 ), ( 21, 22 ) )
  vector.print %w1_2 : vector<2x2xi32>


  // NEED-IMPL: strides are used to specify how sub-vector are inserted with steps
  // Currently strides only allow 1s, so the example above could not work yet.
  // when offset = [0, 0], strides =
  // [2, 1]:  x x o o    | [1, 2]:  x o x o   | [2, 2]: x o x o
  //          o o o o    |          x o x o   |         o o o o
  //          x x o o    |          o o o o   |         x o x o
  //          o o o o    |          o o o o   |         o o o o

  // %w2_0 = vector.extract_strided_slice %base { offsets = [0, 0], sizes = [2, 2], strides = [2, 1] }
  //   : vector<4x4xi32> to vector<2x2xi32>
  // %w2_1 = vector.extract_strided_slice %base { offsets = [0, 0], sizes = [2, 2], strides = [1, 2] }
  //   : vector<4x4xi32> to vector<2x2xi32>
  // %w2_2 = vector.extract_strided_slice %base { offsets = [0, 0], sizes = [2, 2], strides = [2, 2] }
  //   : vector<4x4xi32> to vector<2x2xi32>

  // vector.print %w2_0 : vector<2x2xi32>
  // vector.print %w2_1 : vector<2x2xi32>
  // vector.print %w2_2 : vector<2x2xi32>


  // vector.extract_strided_slice with any rank can be defined recursively:

  // vector.extract_strided_slice %b { offsets = [o0, ...], sizes = [l0, ...], strides = [s0, ...] }
  // <==>
  // [
  //    vector.extract_strided_slice %b[o0 + 0*s0] { offsets = [...], sizes = [...], strides = [...] },
  //    vector.extract_strided_slice %b[o0 + 1*s0] { offsets = [...], sizes = [...], strides = [...] },
  //    vector.extract_strided_slice %b[o0 + 2*s0] { offsets = [...], sizes = [...], strides = [...] },
  //    ...
  //    vector.extract_strided_slice %b[o0 + (l0-1)*s0] { offsets = [...], sizes = [...], strides = [...] }
  // ]

  %big_base = arith.constant dense<[
    [[0, 10, 20, 30], [100, 110, 120, 130], [200, 210, 220, 230], [300, 310, 320, 330]],
    [[1, 11, 21, 31], [101, 111, 121, 131], [201, 211, 221, 231], [301, 311, 321, 331]],
    [[2, 12, 22, 32], [102, 112, 122, 132], [202, 212, 222, 232], [302, 312, 322, 332]],
    [[3, 13, 23, 33], [103, 113, 123, 133], [203, 213, 223, 233], [303, 313, 323, 333]]]>
  : vector<4x4x4xi32>


  %w3 = vector.extract_strided_slice %big_base
    { offsets = [1, 0, 0], sizes = [2, 3, 3], strides = [1, 1, 1] }
    : vector<4x4x4xi32> to vector<2x3x3xi32>
  // CHECK: ( ( ( 1, 11, 21 ), ( 101, 111, 121 ), ( 201, 211, 221 ) ),
  // CHECK-SAME: ( ( 2, 12, 22 ), ( 102, 112, 122 ), ( 202, 212, 222 ) ) )
  vector.print %w3 : vector<2x3x3xi32>


  // Note that currently vector.extract_strided_slice do NOT support to extract
  // a lower-rank vector from a bigger one.
  // So if we want a lower-rank sub-vector from a bigger one, we can NOT write this:

  // %w4_0 = vector.extract_strided_slice %big_base
  //   { offsets = [1, 0, 0], sizes = [3, 3], strides = [1, 1] }
  //   : vector<4x4x4xi32> to vector<3x3xi32>

  // Instead, we either first extract %big_base[1], or do an extra extract with result:
  %t1 = vector.extract %big_base[1] : vector<4x4xi32> from vector<4x4x4xi32>
  %w4_1 = vector.extract_strided_slice %t1
    { offsets = [0, 0], sizes = [3, 3], strides = [1, 1] }
    : vector<4x4xi32> to vector<3x3xi32>

  %t2 = vector.extract_strided_slice %big_base
    { offsets = [1, 0, 0], sizes = [1, 3, 3], strides = [1, 1, 1] }
    : vector<4x4x4xi32> to vector<1x3x3xi32>
  %w4_2 = vector.extract %t2[0] : vector<3x3xi32> from vector<1x3x3xi32>
  // CHECK: ( ( 1, 11, 21 ),
  // CHECK-SAME: ( 101, 111, 121 ),
  // CHECK-SAME: ( 201, 211, 221 ) )
  vector.print %w4_1 : vector<3x3xi32>
  // CHECK: ( ( 1, 11, 21 ),
  // CHECK: ( 101, 111, 121 ),
  // CHECK: ( 201, 211, 221 ) )
  vector.print %w4_2 : vector<3x3xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
