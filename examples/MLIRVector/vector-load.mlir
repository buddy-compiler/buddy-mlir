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

memref.global "private" @gv2 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>

func.func @kernel_1(%arg0: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  // load normal usage
  %v0 = vector.load %arg0[%c0] : memref<8xi32>, vector<3xi32>
  // CHECK: ( 0, 1, 2 )
  vector.print %v0 : vector<3xi32>
  return
}

func.func @main() -> i32 {
  // vector.load can load n-D vector from m-D scalar memref or k-D vector memref

  // preparation for examples
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %base0 = memref.get_global @gv0 : memref<8xi32>
  %base1 = memref.get_global @gv1 : memref<4x4xi32>
  %base2 = memref.get_global @gv2 : memref<8xi32>

  call @kernel_1(%base0) : (memref<8xi32>) -> ()

  // load with m-D memref
  //  case 1: inside inner-most dimension
  %v1 = vector.load %base1[%c1, %c0] : memref<4x4xi32>, vector<4xi32>
  // CHECK: ( 4, 5, 6, 7 )
  vector.print %v1 : vector<4xi32>


  // load with m-D memref
  //  case 2 : cross inner-most dimension
  // In this case, it will behavior like the memref is "flat"
  %v2 = vector.load %base1[%c1, %c1] : memref<4x4xi32>, vector<4xi32>
  // ( 5, 6, 7, 8 )
  vector.print %v2 : vector<4xi32>


  // The shape of vector can be strange.
  // TODO: figure out why it failed. The document says it SHOULD work.

  // %v3 = vector.load %base1[%c1, %c1] : memref<4x4xi32>, vector<3x3xi32>
  // vector.print %v3 : vector<3x3xi32>


  // load with memref of vector
  // prepare for the memref:
  // FIXME: use literal form instead of vector.store
  %base4 = memref.alloc() : memref<2xvector<4xi32>>
  %w0 = arith.constant dense<[100, 101, 102, 103]> : vector<4xi32>
  %w1 = arith.constant dense<[104, 105, 106, 107]> : vector<4xi32>
  vector.store %w0, %base4[%c0] : memref<2xvector<4xi32>>, vector<4xi32>
  vector.store %w1, %base4[%c1] : memref<2xvector<4xi32>>, vector<4xi32>

  %v4 = vector.load %base4[%c0] : memref<2xvector<4xi32>>, vector<4xi32>
  // ( 100, 101, 102, 103 )
  vector.print %v4 : vector<4xi32>

  // This one fail. The shape of result must be exactly the element vector shape.
  // %v4_1 = vector.load %base4[%c0] : memref<2xvector<4xi32>>, vector<2x4xi32>


  // load with dynamic memref
  //    case 1: in-bound
  %base5 = memref.cast %base1 : memref<4x4xi32> to memref<?x?xi32>
  %v5 = vector.load %base5[%c1, %c1] : memref<?x?xi32>, vector<8xi32>
  // ( 5, 6, 7, 8, 9, 10, 11, 12 )
  vector.print %v5 : vector<8xi32>


  // load with dynamic memref
  //    case 2: out of bound
  // The document says:
  //    Representation-wise, the ‘vector.load’ operation permits out-of-bounds reads.
  //    Support and implementation of out-of-bounds vector loads is target-specific.
  //    No assumptions should be made on the value of elements loaded out of bounds.
  //    Not all targets may support out-of-bounds vector loads.
  %v6 = vector.load %base5[%c3, %c1] : memref<?x?xi32>, vector<8xi32>
  // ( 13, 14, 15, 0, 1, 2, 3, 4 )
  vector.print %v6 : vector<8xi32>


  // load with unranked memref is not allowed
  %base6 = memref.cast %base1 : memref<4x4xi32> to memref<*xi32>
  // %v7 = vector.load %base6[%c0, %c0] : memref<*xi32>, vector<8xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
