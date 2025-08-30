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

memref.global "private" @gv0 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>

memref.global "private" @gv1 : memref<4x4xi32> = dense<[[0, 1, 2, 3],
                                                        [4, 5, 6, 7],
                                                        [8, 9, 10, 11],
                                                        [12, 13, 14, 15]]>
memref.global "private" @gv2 : memref<8xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7]>

func.func private @printMemrefI32(memref<*xi32>)

func.func @main() -> i32 {
  // expandload is also a load with a mask, but it moves to read the next
  // element in memory only when the mask at the current lane is on, meaning:
  //    result[0] := mask[0] ? base[index++] : pass_thru[0]
  //    result[1] := mask[1] ? base[index++] : pass_thru[1]
  //    ...
  // instead of (This is what vector.maskedload does)
  //    result[0] := mask[0] ? base[0] : pass_thru[0]
  //    result[1] := mask[1] ? base[1] : pass_thru[1]
  //    ...

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

  // expandload normal usage
  // expandload requires a pass-through value at any time
  %mask0 = arith.constant dense<[1, 0, 1, 0]> : vector<4xi1>

  // %v0 will be [0, 2331, 1, 2333] instead of [0, 2331, 2, 2333]
  // because lane 1 is masked off, it will not move to load the next element
  // in memory. So at lane 2, it still loads i32 1 at memory instead of 2.
  %v0 = vector.expandload %base0[%c0], %mask0, %pass_thru_4
    : memref<8xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>

  // CHECK: ( 0, 2331, 1, 2333 )
  vector.print %v0 : vector<4xi32>


  // expandload with multi-dimension memref
  //    case 1: inside most-inner dimension
  %mask1 = arith.constant dense<[1, 0, 0, 1]> : vector<4xi1>

  %v1 = vector.expandload %base1[%c0, %c0], %mask1, %pass_thru_4
    : memref<4x4xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
  // CHECK: ( 0, 2331, 2332, 1 )
  vector.print %v1 : vector<4xi32>


  // expandload with multi-dimension memref
  //    case 2: cross the most-inner dimension
  // In this case, it will behave like the memref is flat
  %mask2 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>

  %v2 = vector.expandload %base1[%c0, %c0], %mask2, %pass_thru_8
    : memref<4x4xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
  // CHECK: ( 0, 2331, 1, 2, 3, 4, 2336, 2337 )
  vector.print %v2 : vector<8xi32>


  // expandload with memref with custom layout
  // TODO: find out how to create a memref with arbitrarily affine map layout
  // "3" is reserved for this example


  // expandload with dynamic memref
  //    case 1: in-bound
  %base4 = memref.cast %base1 : memref<4x4xi32> to memref<?x?xi32>
  %mask4 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>

  %v4 = vector.expandload %base4[%c1, %c1], %mask4, %pass_thru_8
    : memref<?x?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
  // CHECK: ( 5, 2331, 6, 7, 8, 9, 2336, 2337 )
  vector.print %v4 : vector<8xi32>


  // expandload with dynamic memref
  //    case 2: out-of-bound
  // its behavior likes vector.load -- it's a platform-specific operation
  // On some platforms, it will just load data in @gv2 when access is out-of-bound
  %base5 = memref.cast %base1 : memref<4x4xi32> to memref<?x?xi32>
  %mask5 = arith.constant dense<[1, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>

  %v5 = vector.expandload %base5[%c3, %c1], %mask5, %pass_thru_8
    : memref<?x?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
  // CHECK: ( 13, 2331, 14, 15, 0, 1, 2336, 2337 )
  vector.print %v5 : vector<8xi32>


  // expandload with unranked memref is not allowed
  // %base6 = memref.cast %base1 : memref<4x4xi32> to memref<*xi32>
  // %mask6 = arith.constant dense<[1, 0, 0, 1]> : vector<4xi1>

  // %v6 = vector.expandload %base6[%c0, %c0], %mask6, %pass_thru_4
  //   : memref<*xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
