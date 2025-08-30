// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -split-input-file -verify-diagnostics \
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
  // vector.gather is also a load with mask, but it load elements in a custom order,
  // rather than sequentially:
  //    result[0] := mask[0] ? base[index[0]] : pass_thru[0]
  //    result[1] := mask[1] ? base[index[1]] : pass_thru[1]
  //    ...
  // As a comparison, that's what vector.maskedload does:
  //    result[0] := mask[0] ? base[0] : pass_thru[0]
  //    result[1] := mask[1] ? base[1] : pass_thru[1]
  //    ...
  // vector.gather supports loading n-D vector.

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
  %pass_thru_2x2 = arith.constant dense<233> : vector<2x2xi32>

  // gather normal usage
  %mask0 = arith.constant dense<1> : vector<4xi1>
  %index0 = arith.constant dense<[3, 4, 2, 1]> : vector<4xi32>

  %v0 = vector.gather %base0[%c0][%index0], %mask0, %pass_thru_4
    : memref<8xi32>, vector<4xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
  // CHECK: ( 3, 4, 2, 1 )
  vector.print %v0 : vector<4xi32>


  // with mask
  %mask1 = arith.constant dense<[1, 0, 1, 0]> : vector<4xi1>
  %index1 = arith.constant dense<[3, 4, 2, 1]> : vector<4xi32>

  %v1 = vector.gather %base0[%c0][%index1], %mask1, %pass_thru_4
    : memref<8xi32>, vector<4xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
  // CHECK: ( 3, 2331, 2, 2333 )
  vector.print %v1 : vector<4xi32>


  // for n-D vectors, the element is numbered as if they are in flat memory
  // In this example, we load a 2x2 vector at the center of a 4x4 memory, and
  // they are numbered like the right-hand side:
  //  o o o o   |   o o o o
  //  o x x o   |   o 1 2 o
  //  o x x o   |   o 3 4 o
  //  o o o o   |   o o o o
  %mask2 = arith.constant dense<1> : vector<2x2xi1>
  %index2 = arith.constant dense<[[3, 4], [2, 1]]> : vector<2x2xi32>

  %v2 = vector.gather %base1[%c1, %c1][%index2], %mask2, %pass_thru_2x2
    : memref<4x4xi32>, vector<2x2xi32>, vector<2x2xi1>, vector<2x2xi32> into vector<2x2xi32>
  // CHECK: ( ( 8, 9 ), ( 7, 6 ) )
  vector.print %v2 : vector<2x2xi32>


  // For the same reason as vector.load, the indices can be negative or out-of-bound,
  // and the behavior will be specified by platforms.
  %mask3 = arith.constant dense<1> : vector<2x2xi1>
  %index3 = arith.constant dense<[[-1, -8], [5, 13]]> : vector<2x2xi32>

  %v3 = vector.gather %base1[%c1, %c1][%index3], %mask3, %pass_thru_2x2
    : memref<4x4xi32>, vector<2x2xi32>, vector<2x2xi1>, vector<2x2xi32> into vector<2x2xi32>
  // CHECK: ( ( 4, 5 ), ( 10, 2 ) )
  vector.print %v3 : vector<2x2xi32>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
