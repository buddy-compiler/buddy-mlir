// RUN: buddy-opt %s --lower-linalg-to-ame='target=gem5' | FileCheck %s
// RUN: buddy-opt %s --lower-linalg-to-ame | FileCheck %s --check-prefix=PROFILE8

#id = affine_map<(m, n) -> (m, n)>
#transpose = affine_map<(m, n) -> (n, m)>
#row = affine_map<(m, n) -> (n)>
#column = affine_map<(m, n) -> (m)>
#scalar = affine_map<(m, n) -> ()>
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

// A generic contraction with the standard i32 wrapping multiply/add body has
// the same contract as the named matmul. The add and multiply are commuted to
// check SSA operands rather than relying on textual operand order.
func.func @generic_matmul(%a: memref<5x7xi32>, %b: memref<7x6xi32>,
                          %c: memref<5x6xi32>) {
  linalg.generic {
    indexing_maps = [#lhs, #rhs, #out],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%a, %b : memref<5x7xi32>, memref<7x6xi32>)
    outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %product = arith.muli %y, %x : i32
    %sum = arith.addi %initial, %product : i32
    linalg.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_matmul(
// CHECK-NOT: linalg.generic
// CHECK: ame.acquire
// CHECK: !llvm.target<"riscv.ztt.matrix", i32, 4, 4>
// CHECK: ame.mmulacc.2d
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return
// PROFILE8-LABEL: func.func @generic_matmul(
// PROFILE8: !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
// PROFILE8: ame.mmulacc.2d

// Elementwise arithmetic keeps nonzero offsets and non-unit input strides by
// packing through memref operations. Output dimensions supply the dynamic
// iteration space, and the final tile is masked on both axes.
func.func @generic_add(
    %a: memref<?x?xi32, strided<[?, ?], offset: ?>>,
    %b: memref<?x?xi32, strided<[?, ?], offset: ?>>,
    %c: memref<?x?xi32, strided<[?, ?], offset: ?>>) {
  linalg.generic {
    indexing_maps = [#id, #id, #id],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %b : memref<?x?xi32, strided<[?, ?], offset: ?>>,
                    memref<?x?xi32, strided<[?, ?], offset: ?>>)
    outs(%c : memref<?x?xi32, strided<[?, ?], offset: ?>>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %sum = arith.addi %x, %y : i32
    linalg.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_add(
// CHECK-NOT: linalg.generic
// CHECK: ame.acquire
// CHECK: memref.load
// CHECK: ame.madd.ew
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK: } else {
// CHECK-NOT: ame.
// CHECK: arith.addi
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_sub(%a: memref<5x6xi32>, %b: memref<5x6xi32>,
                       %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<5x6xi32>, memref<5x6xi32>)
      outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %difference = arith.subi %x, %y : i32
    linalg.yield %difference : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_sub(
// CHECK-NOT: linalg.generic
// CHECK: ame.msub.ew
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_mul(%a: memref<5x6xi32>, %b: memref<5x6xi32>,
                       %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<5x6xi32>, memref<5x6xi32>)
      outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %product = arith.muli %x, %y : i32
    linalg.yield %product : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_mul(
// CHECK-NOT: linalg.generic
// CHECK: ame.mmul.ew
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_bitwise(%a: memref<5x6xi32>, %b: memref<5x6xi32>,
                           %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<5x6xi32>, memref<5x6xi32>)
      outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %and = arith.andi %x, %y : i32
    %or = arith.ori %and, %initial : i32
    %xor = arith.xori %or, %x : i32
    linalg.yield %xor : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_bitwise(
// CHECK-NOT: linalg.generic
// CHECK: ame.mand.ew
// CHECK: ame.mor.ew
// CHECK: ame.mxor.ew
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_minmax(%a: memref<5x6xi32>, %b: memref<5x6xi32>,
                          %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<5x6xi32>, memref<5x6xi32>)
      outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %min = arith.minsi %x, %y : i32
    %max = arith.maxsi %min, %initial : i32
    linalg.yield %max : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_minmax(
// CHECK-NOT: linalg.generic
// CHECK: ame.mmin.ew
// CHECK: ame.mmax.ew
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_copy(%a: memref<5x6xi32>, %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<5x6xi32>) outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_copy(
// CHECK-NOT: linalg.generic
// CHECK: ame.mls.rm
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_transpose(%a: memref<6x5xi32>, %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#transpose, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<6x5xi32>) outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_transpose(
// CHECK-NOT: linalg.generic
// CHECK: ame.mls.cm
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_row_broadcast(%a: memref<6xi32>, %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#row, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<6xi32>) outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_row_broadcast(
// CHECK-NOT: linalg.generic
// CHECK: ame.mrowbcast.ew.x
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_column_broadcast(%a: memref<5xi32>, %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#column, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<5xi32>) outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_column_broadcast(
// CHECK-NOT: linalg.generic
// CHECK: ame.mcolbcast.ew.x
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_scalar_broadcast(%a: memref<i32>, %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#scalar, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<i32>) outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_scalar_broadcast(
// CHECK-NOT: linalg.generic
// CHECK: ame.mbcast.m.x
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @generic_shifts(%a: memref<5x6xi32>, %b: memref<5x6xi32>,
                          %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<5x6xi32>, memref<5x6xi32>)
      outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %left = arith.shli %x, %y : i32
    %signed = arith.shrsi %left, %y : i32
    %unsigned = arith.shrui %signed, %y : i32
    linalg.yield %unsigned : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_shifts(
// CHECK-NOT: linalg.generic
// CHECK: ame.msll.ew
// CHECK: ame.msra.ew
// CHECK: ame.msrl.ew
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

// Unsigned comparisons reinterpret i32 bits with a distinct matrix datatype;
// their results can subsequently participate in signed arithmetic.
func.func @generic_unsigned_minmax(%a: memref<5x6xi32>, %b: memref<5x6xi32>,
                                   %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<5x6xi32>, memref<5x6xi32>)
      outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %min = arith.minui %x, %y : i32
    %max = arith.maxui %min, %initial : i32
    %sum = arith.addi %max, %x : i32
    linalg.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_unsigned_minmax(
// CHECK-NOT: linalg.generic
// CHECK: arith.constant 32 : i64
// CHECK: ame.mmin.ew
// CHECK: ame.mmax.ew
// CHECK: ame.madd.ew
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

// One-dimensional regions use one useful column and retain their tail.
func.func @generic_rank_one(%a: memref<?xi32>, %c: memref<?xi32>,
                            %scale: i32) {
  linalg.generic {indexing_maps = [affine_map<(m) -> (m)>,
                                   affine_map<(m) -> (m)>],
                  iterator_types = ["parallel"]}
      ins(%a : memref<?xi32>) outs(%c : memref<?xi32>) {
  ^bb0(%x: i32, %initial: i32):
    %constant = arith.constant -3 : i32
    %scaled = arith.muli %x, %scale : i32
    %sum = arith.addi %scaled, %constant : i32
    linalg.yield %sum : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_rank_one(
// CHECK-NOT: linalg.generic
// CHECK: ame.mbcast.m.x
// CHECK: ame.mmul.ew
// CHECK: ame.madd.ew
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

// Scalar ins operands use an empty map, just like rank-zero memrefs, but do
// not perform a memory read. Scalar captures elsewhere remain supported too.
func.func @generic_scalar_input(%a: memref<5x6xi32>, %factor: i32,
                                %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #scalar, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %factor : memref<5x6xi32>, i32)
      outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %scale: i32, %initial: i32):
    %scaled = arith.muli %x, %scale : i32
    linalg.yield %scaled : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_scalar_input(
// CHECK-NOT: linalg.generic
// CHECK: ame.mbcast.m.x
// CHECK: ame.mmul.ew
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

// Input maps are composed with the inverse of the output map. Output loop
// coordinates can therefore be permuted without transposing the result twice.
func.func @generic_permuted_output(%a: memref<6x5xi32>,
                                   %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #transpose],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<6x5xi32>) outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_permuted_output(
// CHECK-NOT: linalg.generic
// CHECK: ame.mls.cm
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

// A supported subexpression is insufficient: replacing only its producer
// would silently lose the division. The complete generic must be retained.
func.func @unsupported_division(%a: memref<4x4xi32>, %b: memref<4x4xi32>,
                                %c: memref<4x4xi32>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<4x4xi32>, memref<4x4xi32>)
      outs(%c : memref<4x4xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %sum = arith.addi %x, %y : i32
    %div = arith.divsi %sum, %y : i32
    linalg.yield %div : i32
  }
  return
}
// CHECK-LABEL: func.func @unsupported_division(
// CHECK-NOT: ame.
// CHECK: linalg.generic
// CHECK: arith.divsi
// CHECK-NOT: ame.
// CHECK: return

func.func @unsupported_f32_generic(%a: memref<4x4xf32>,
                                  %b: memref<4x4xf32>,
                                  %c: memref<4x4xf32>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
      outs(%c : memref<4x4xf32>) {
  ^bb0(%x: f32, %y: f32, %initial: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
  }
  return
}
// CHECK-LABEL: func.func @unsupported_f32_generic(
// CHECK-NOT: ame.
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK-NOT: ame.
// CHECK: return

func.func @unsupported_tensor_generic(%a: tensor<4x4xi32>,
                                     %b: tensor<4x4xi32>,
                                     %c: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %result = linalg.generic {indexing_maps = [#id, #id, #id],
                            iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<4x4xi32>, tensor<4x4xi32>)
      outs(%c : tensor<4x4xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %sum = arith.addi %x, %y : i32
    linalg.yield %sum : i32
  } -> tensor<4x4xi32>
  return %result : tensor<4x4xi32>
}
// CHECK-LABEL: func.func @unsupported_tensor_generic(
// CHECK-NOT: ame.
// CHECK: linalg.generic
// CHECK-NOT: ame.
// CHECK: return

// The contraction shape alone does not imply a matmul: this body subtracts C
// from the product and cannot be replaced by mmulacc.2d.
func.func @unsupported_contraction_body(%a: memref<4x4xi32>,
                                       %b: memref<4x4xi32>,
                                       %c: memref<4x4xi32>) {
  linalg.generic {indexing_maps = [#lhs, #rhs, #out],
                  iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<4x4xi32>, memref<4x4xi32>)
      outs(%c : memref<4x4xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %product = arith.muli %x, %y : i32
    %difference = arith.subi %product, %initial : i32
    linalg.yield %difference : i32
  }
  return
}
// CHECK-LABEL: func.func @unsupported_contraction_body(
// CHECK-NOT: ame.
// CHECK: linalg.generic
// CHECK: arith.subi
// CHECK-NOT: ame.
// CHECK: return

func.func @generic_fill(%c: memref<5x6xi32>, %value: i32) {
  linalg.generic {indexing_maps = [#id],
                  iterator_types = ["parallel", "parallel"]}
      outs(%c : memref<5x6xi32>) {
  ^bb0(%initial: i32):
    linalg.yield %value : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_fill(
// CHECK-NOT: linalg.generic
// CHECK-NOT: memref.load
// CHECK: ame.mbcast.m.x
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

// The loop dimensions are not inherently rows then columns. Normalization
// against the output permutation makes both these transposes cancel.
func.func @generic_cancelled_permutations(%a: memref<5x6xi32>, %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#transpose, #transpose],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<5x6xi32>) outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}
// CHECK-LABEL: func.func @generic_cancelled_permutations(
// CHECK-NOT: linalg.generic
// CHECK: ame.mls.rm
// CHECK-NOT: ame.mls.cm
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK-NOT: linalg.generic
// CHECK: return

func.func @unsupported_index_body(%c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id],
                  iterator_types = ["parallel", "parallel"]}
      outs(%c : memref<5x6xi32>) {
  ^bb0(%initial: i32):
    %index = linalg.index 0 : index
    %value = arith.index_cast %index : index to i32
    linalg.yield %value : i32
  }
  return
}
// CHECK-LABEL: func.func @unsupported_index_body(
// CHECK-NOT: ame.
// CHECK: linalg.generic
// CHECK: linalg.index
// CHECK-NOT: ame.
// CHECK: return

func.func @unsupported_multiple_outputs(%a: memref<5x6xi32>,
                                       %b: memref<5x6xi32>,
                                       %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<5x6xi32>) outs(%b, %c : memref<5x6xi32>, memref<5x6xi32>) {
  ^bb0(%x: i32, %b_old: i32, %c_old: i32):
    linalg.yield %x, %x : i32, i32
  }
  return
}
// CHECK-LABEL: func.func @unsupported_multiple_outputs(
// CHECK-NOT: ame.
// CHECK: linalg.generic
// CHECK-NOT: ame.
// CHECK: return

func.func @unsupported_memory_space(%a: memref<5x6xi32, "device">,
                                   %c: memref<5x6xi32, "device">) {
  linalg.generic {indexing_maps = [#id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<5x6xi32, "device">)
      outs(%c : memref<5x6xi32, "device">) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}
// CHECK-LABEL: func.func @unsupported_memory_space(
// CHECK-NOT: ame.
// CHECK: linalg.generic
// CHECK-NOT: ame.
// CHECK: return
