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

memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f1 = arith.constant 1.0 : f32

  // Quick examples:
  // Broadcast scalar to 1-D vector.
  %ele = memref.load %mem[%c1, %c1] : memref<4x4xf32>
  %broadcast_vec = vector.broadcast %ele : f32 to vector<4xf32>
  // CHECK: ( 11, 11, 11, 11 )
  vector.print %broadcast_vec : vector<4xf32>

  // Broadcast 1-D vector to 2-D vector.
  %load_vec = vector.load %mem[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
  %broadcast_vec_2d = vector.broadcast %load_vec : vector<4xf32> to vector<4x4xf32>
  // CHECK: ( ( 0, 1, 2, 3 ), ( 0, 1, 2, 3 ), ( 0, 1, 2, 3 ), ( 0, 1, 2, 3 ) )
  vector.print %broadcast_vec_2d : vector<4x4xf32>


  // Detailed examples for all cases of vector.broadcast:
  %v1 = arith.constant dense<[[1.0], [2.0], [3.0]]> : vector<3x1xf32>
  %v2 = arith.constant dense<[[[1.0], [2.0], [3.0]]]> : vector<1x3x1xf32>
  // CHECK: ( ( 1 ), ( 1 ), ( 1 ) )
  // CHECK-NEXT: ( ( 1 ), ( 1 ), ( 1 ) )
  func.call @broadcast_scalar_to_vector(%f1) : (f32) -> vector<3x1xf32>
  // CHECK:    ( ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECk-SAME: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ) )
  // CHECK-NEXT: ( ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME:   ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME:   ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME:   ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ) )
  // CHECK-NEXT:    ( ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECk-SAME: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ) )
  // CHECK-NEXT: ( ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME:   ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME:   ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME:   ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ) )
  func.call @broadcast_low_dim_to_high_dim(%v1) : (vector<3x1xf32>) -> vector<4x3x2xf32>
  // CHECK:    ( ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECk-SAME: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ) )
  // CHECK-NEXT: ( ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME:   ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME:   ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ),
  // CHECK-SAME:   ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) ) )
  func.call @broadcast_1_to_n_case(%v2) : (vector<1x3x1xf32>) -> vector<4x3x2xf32>
  // CHECK: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) )
  // CHECK: ( ( 1, 1 ), ( 2, 2 ), ( 3, 3 ) )
  func.call @broadcast_n_to_n_case(%v1) : (vector<3x1xf32>) -> vector<3x2xf32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}

func.func @broadcast_scalar_to_vector(%src: f32) -> vector<3x1xf32> {
  // broadcast a scalar to vector
  %result = vector.broadcast %src : f32 to vector<3x1xf32>

  // equals to splat it, like "std::fill" in C++.
  // %w will be [[%src], [%src], [%src]]
  %w = vector.splat %src : vector<3x1xf32>

  // %w == %result
  vector.print %result : vector<3x1xf32>
  vector.print %w : vector<3x1xf32>

  return %result : vector<3x1xf32>
}

func.func @broadcast_low_dim_to_high_dim(%src: vector<3x1xf32>) -> vector<4x3x2xf32> {
  // broadcasting vector with smaller rank to larger one can be transformed into
  // broadcasting two vectors with the same rank
  %result = vector.broadcast %src : vector<3x1xf32> to vector<4x3x2xf32>

  // We can extend a vector to any higher dimension by adding dimensions with
  // length "1" in front of it. For example, extend vector<3x1xf32> to
  // vector<1x3x1xf32>, or vector<1x1x1x1x3x1xf32> if we need.
  %zero = arith.constant 0.0 : f32
  %t0 = vector.broadcast %zero : f32 to vector<1x3x1xf32>
  %t1 = vector.insert %src, %t0[0] : vector<3x1xf32> into vector<1x3x1xf32>

  // Then we need to broadcast two rank-equal vectors, which can be done
  // recursively on each dimension. Please check @broadcast_1_to_n_cast for details.
  %t2 = func.call @broadcast_1_to_n_case(%t1) : (vector<1x3x1xf32>) -> vector<4x3x2xf32>

  // %result == %t2
  vector.print %result : vector<4x3x2xf32>
  vector.print %t2 : vector<4x3x2xf32>

  return %result : vector<4x3x2xf32>
}

func.func @broadcast_1_to_n_case(%src: vector<1x3x1xf32>) -> vector<4x3x2xf32> {
  // Case 1, "1->n" kind.
  //    broadcast %src: vector<1 x ... x T> to vector<n x ... x T>
  %result = vector.broadcast %src : vector<1x3x1xf32> to vector<4x3x2xf32>

  // let %src == [%e0], then the new vector will be:
  //    [broadcast %sub : vector<... x T>, ..., broadcast %sub : vector<... x T>]

  // for example, here we get %src = [%e0], n = 4
  %e0 = vector.extract %src[0] : vector<3x1xf32> from vector<1x3x1xf32>

  // then make %t0, %t1, %t2, %t3 to be broadcast %e0 : vector<... x T>
  %t0 = vector.broadcast %e0 : vector<3x1xf32> to vector<3x2xf32>
  %t1 = vector.broadcast %e0 : vector<3x1xf32> to vector<3x2xf32>
  %t2 = vector.broadcast %e0 : vector<3x1xf32> to vector<3x2xf32>
  %t3 = vector.broadcast %e0 : vector<3x1xf32> to vector<3x2xf32>

  // then create vector as [%t0, %t1, %t2, %t3]
  %zero = arith.constant 0.0 : f32
  %w_ = vector.broadcast %zero : f32 to vector<4x3x2xf32>
  %w0 = vector.insert %t0, %w_[0] : vector<3x2xf32> into vector<4x3x2xf32>
  %w1 = vector.insert %t1, %w0[1] : vector<3x2xf32> into vector<4x3x2xf32>
  %w2 = vector.insert %t2, %w1[2] : vector<3x2xf32> into vector<4x3x2xf32>
  %w3 = vector.insert %t3, %w2[3] : vector<3x2xf32> into vector<4x3x2xf32>

  // now the final result %w3 will equal to %result
  vector.print %result : vector<4x3x2xf32>
  vector.print %w3 : vector<4x3x2xf32>

  return %result : vector<4x3x2xf32>
}

func.func @broadcast_n_to_n_case(%src: vector<3x1xf32>) -> vector<3x2xf32> {
  // Case 2, "n->n" kind.
  //    broadcast %src: vector<n x ... x T> to vector<n x ... x T>
  %result = vector.broadcast %src : vector<3x1xf32> to vector<3x2xf32>

  // let %src == [%e0, %e1, ..., %e_{n-1}], then the new vector will be:
  //  [
  //     broadcast %e0 : vector<... x T>,
  //     broadcast %e1 : vector<... x T>,
  //     ...,
  //     broadcast %e_{n-1} : vector<... x T>
  //  ]

  // get elements
  %e0 = vector.extract %src[0] : vector<1xf32> from vector<3x1xf32>
  %e1 = vector.extract %src[1] : vector<1xf32> from vector<3x1xf32>
  %e2 = vector.extract %src[2] : vector<1xf32> from vector<3x1xf32>

  // broadcast with elements
  %t0 = vector.broadcast %e0 : vector<1xf32> to vector<2xf32>
  %t1 = vector.broadcast %e1 : vector<1xf32> to vector<2xf32>
  %t2 = vector.broadcast %e2 : vector<1xf32> to vector<2xf32>

  // construct [%t0, %t1, %t2]
  %zero = arith.constant 0.0 : f32
  %w_ = vector.broadcast %zero : f32 to vector<3x2xf32>
  %w0 = vector.insert %t0, %w_[0] : vector<2xf32> into vector<3x2xf32>
  %w1 = vector.insert %t1, %w0[1] : vector<2xf32> into vector<3x2xf32>
  %w2 = vector.insert %t2, %w1[2] : vector<2xf32> into vector<3x2xf32>

  // now %w2 == %result
  vector.print %result : vector<3x2xf32>
  vector.print %w2 : vector<3x2xf32>

  return %result : vector<3x2xf32>
}
