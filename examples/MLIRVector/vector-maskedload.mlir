// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
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
memref.global "private" @gv2 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @main() -> i32 {
  // maskedload is a load with a mask, supporting to load
  // an 1-D vector into a n-D memref with the mask.

  // preparation for examples
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %base0 = memref.get_global @gv0 : memref<8xi32>
  %base1 = memref.get_global @gv1 : memref<4x4xi32>
  %base2 = memref.get_global @gv2 : memref<8xi32>

  %pass_thru_4 = arith.constant dense<[2330, 2331, 2332, 2333]> : vector<4xi32>
  %pass_thru_8 = arith.constant dense<[2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337]> : vector<8xi32>

  // maskedload normal usage
  // maskedload requires a pass-through value at any time
  %mask0 = arith.constant dense<[1, 0, 1, 0]> : vector<4xi1>
  %v0 = vector.maskedload %base0[%c0], %mask0, %pass_thru_4
    : memref<8xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>


  // maskedload with multi-dimension memref
  //    case 1: inside most-inner dimension
  %mask1 = arith.constant dense<[1, 0, 0, 1]> : vector<4xi1>

  %v1 = vector.maskedload %base1[%c0, %c0], %mask1, %pass_thru_4
    : memref<4x4xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
  // CHECK: ( 0, 2331, 2332, 3 )
  vector.print %v1 : vector<4xi32>


  // maskedload with multi-dimension memref
  //    case 2: cross the most-inner dimension
  // In this case, it will behave like the memref is flat
  %mask2 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>

  %v2 = vector.maskedload %base1[%c0, %c0], %mask2, %pass_thru_8
    : memref<4x4xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
  // CHECK: ( 0, 2331, 2, 3, 4, 5, 2336, 2337 )
  vector.print %v2 : vector<8xi32>


  // maskedload with memref with custom layout
  // TODO: find out how to create a memref with arbitrarily affine map layout
  // "3" is reserved for this example


  // maskedload with dynamic memref
  //    case 1: in-bound
  %base4 = memref.cast %base1 : memref<4x4xi32> to memref<?x?xi32>
  %mask4 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>

  %v4 = vector.maskedload %base4[%c1, %c1], %mask4, %pass_thru_8
    : memref<?x?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
  // CHECK: ( 5, 2331, 7, 8, 9, 10, 2336, 2337 )
  vector.print %v4 : vector<8xi32>


  // maskedload with dynamic memref
  //    case 2: out-of-bound
  // like vector.load, it's a platform-specific operation
  // On some platforms, it will just load data in @gv2 when access is out-of-bound
  %base5 = memref.cast %base1 : memref<4x4xi32> to memref<?x?xi32>
  %mask5 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>

  %v5 = vector.maskedload %base5[%c3, %c1], %mask5, %pass_thru_8
    : memref<?x?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
  // CHECK: ( 13, 2331, 15, 0, 1, 2, 2336, 2337 )
  vector.print %v5 : vector<8xi32>


  // maskedload with unranked memref is not allowed
  // %base6 = memref.cast %base1 : memref<4x4xi32> to memref<*xi32>
  // %mask6 = arith.constant dense<[1, 0, 0, 1]> : vector<4xi1>

  // %v6 = vector.maskedload %base6[%c0, %c0], %mask6, %pass_thru_4
  //   : memref<*xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
