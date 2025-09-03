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

// `vector.bitcast` creates an vector from one type to another type, with the
// exactly same bit pattern.

func.func @main() -> i32 {
  // vector.bitcast cast vector<...x a x T> to vector<... x b x U>, where
  // a * sizeof(T) == b * sizeof(U).
  // Can only deal with innermost dim.

  %v0 = arith.constant dense<[10, 20, 56, 90, 12, 90]> : vector<6xi32>
  // CHECK: ( 10, 20, 56, 90, 12, 90 )
  vector.print %v0 : vector<6xi32>

  // bitcast can change the element type and dimension.
  %v1 = vector.bitcast %v0 : vector<6xi32> to vector<3xi64>
  // CHECK: ( 85899345930, 386547056696, 386547056652 )
  vector.print %v1 : vector<3xi64>

  // it can even change element type from integer to float
  // note that it will preserve bit pattern instead of value.
  %v2 = vector.bitcast %v0 : vector<6xi32> to vector<6xf32>
  // CHECK: ( 1.4013e-44, 2.8026e-44, 7.84727e-44, 1.26117e-43, 1.68156e-44, 1.26117e-43 )
  vector.print %v2 : vector<6xf32>

  // cast it back, and it will be the same vector with exactly
  // every bit same as %v0.
  %v3 = vector.bitcast %v2 : vector<6xf32> to vector<6xi32>
  // CHECk: ( 10, 20, 56, 90, 12, 90 )
  vector.print %v3 : vector<6xi32>

  // bitcast could only be used between vector types with
  // same total length in byte, like 8xi32 <-> 4xf64.

  // error: 'vector.bitcast' op source/result bitwidth of the minor 1-D vectors must be equal
  // %v4 = vector.bitcast %v0 : vector<6xi32> to vector<4xi64>


  // Because scalable vector is platform-specific, vector dialect could not
  // lower or translate them well, so we just assume that we have one:
  //                %v5 : vector<[4]xi32>
  // That's also why we have to comment out the operations below, even if
  // they should be valid usages.


  // bitcast will also accept scalable dimensions
  // %v6 = vector.bitcast %v5 : vector<[4]xi32> to vector<[2]xi64>
  // vector.print %v6 : vector<[2]xi64>

  // %v7 = vector.bitcast %v5 : vector<[4]xi32> to vector<[8]xi16>
  // vector.print %v7 : vector<[8]xi16>


  // bitcast operations of scalable dimensions should ALWAYS meet the
  // bitwidth restriction, not just POSSIBLE to meet it.

  // Both of two operations are not allowed.
  // %v8 = vector.bitcast %v5 : vector<[4]xi32> to vector<[3]xi64> // IMPOSSIBLE to satisfied
  // %v9 = vector.bitcast %v5 : vector<[4]xi32> to vector<[4]xi64> // POSSIBLE to satisfied, but not always

  %ret = arith.constant 0 : i32
  return %ret : i32
}
