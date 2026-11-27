// RUN: buddy-opt %s --lower-linalg-to-ame='target=gem5' | FileCheck %s --check-prefix=AME
// RUN: buddy-opt %s --lower-linalg-to-ame='target=gem5' --lower-ame --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir -o %t.ll
// RUN: buddy-llc %t.ll -mtriple=riscv64 -mattr=+m,+xame,+xame-gem5 -target-abi=lp64 -verify-machineinstrs -filetype=obj -o %t.o
// RUN: buddy-llc %t.ll -mtriple=riscv64 -mattr=+m,+xame,+xame-gem5 -target-abi=lp64 -verify-machineinstrs -o - | FileCheck %s --check-prefix=ASM
// RUN: buddy-opt %s --lower-linalg-to-ame --lower-ame --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir -o %t.generic.ll
// RUN: buddy-llc %t.generic.ll -mtriple=riscv64 -mattr=+m,+xame -target-abi=lp64 -verify-machineinstrs -filetype=obj -o %t.generic.o

// Standalone main for AME_gem5 Linux-SE execution. Every result is checked
// against scalar arithmetic outside linalg.generic. Dimensions cross a 4x4
// tile boundary on both axes. Arithmetic uses dynamic strided memrefs with
// nonzero offsets and reads the original
// destination and includes negative values and wrapping i32 multiplication.
// Both AME acquisition success and its scalar fallback must return zero.

#id = affine_map<(m, n) -> (m, n)>
#transpose = affine_map<(m, n) -> (n, m)>
#row = affine_map<(m, n) -> (n)>
#column = affine_map<(m, n) -> (m)>
#scalar = affine_map<(m, n) -> ()>

// AME-LABEL: func.func @arithmetic(
// AME-NOT: linalg.generic
// AME: ame.madd.ew
// AME: ame.msub.ew
// AME: ame.mmul.ew
// AME: ame.mand.ew
// AME: ame.mor.ew
// AME: ame.mxor.ew
// AME: ame.mmin.ew
// AME: ame.mmax.ew
// AME: ame.release
// AME: return
// ASM-LABEL: arithmetic:
// ASM: madd.ew
// ASM: msub.ew
// ASM: mmul.ew
// ASM: mand.ew
// ASM: mor.ew
// ASM: mxor.ew
// ASM: mmin.ew
// ASM: mmax.ew
// ASM: ame.release
func.func @arithmetic(
    %a: memref<?x?xi32, strided<[?, ?], offset: ?>>,
    %b: memref<?x?xi32, strided<[?, ?], offset: ?>>,
    %c: memref<?x?xi32, strided<[?, ?], offset: ?>>) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<?x?xi32, strided<[?, ?], offset: ?>>,
                    memref<?x?xi32, strided<[?, ?], offset: ?>>)
      outs(%c : memref<?x?xi32, strided<[?, ?], offset: ?>>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %sum = arith.addi %x, %y : i32
    %difference = arith.subi %sum, %initial : i32
    %product = arith.muli %difference, %x : i32
    %and = arith.andi %product, %y : i32
    %or = arith.ori %and, %initial : i32
    %xor = arith.xori %or, %product : i32
    %min = arith.minsi %xor, %x : i32
    %max = arith.maxsi %min, %y : i32
    linalg.yield %max : i32
  }
  return
}

// AME-LABEL: func.func @transpose(
// AME-NOT: linalg.generic
// AME: ame.mls.cm
// AME: ame.release
// AME: return
// ASM-LABEL: transpose:
// ASM: mls.cm
// ASM: ame.release
func.func @transpose(%a: memref<6x5xi32>, %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#transpose, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<6x5xi32>) outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}

// AME-LABEL: func.func @broadcasts(
// AME-NOT: linalg.generic
// AME: ame.mrowbcast.ew.x
// AME: ame.mcolbcast.ew.x
// AME: ame.mbcast.m.x
// AME: ame.release
// AME: return
// ASM-LABEL: broadcasts:
// ASM: mrowbcast.ew.x
// ASM: mcolbcast.ew.x
// ASM: mbcast.m.x
// ASM: ame.release
func.func @broadcasts(%row: memref<6xi32>, %column: memref<5xi32>,
                      %scalar: memref<i32>, %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#row, #column, #scalar, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%row, %column, %scalar : memref<6xi32>, memref<5xi32>, memref<i32>)
      outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %y: i32, %s: i32, %initial: i32):
    %sum = arith.addi %x, %y : i32
    %difference = arith.subi %sum, %s : i32
    linalg.yield %difference : i32
  }
  return
}

// AME-LABEL: func.func @permuted_output(
// AME-NOT: linalg.generic
// AME: ame.mls.cm
// AME: ame.release
// AME: return
func.func @permuted_output(%a: memref<6x5xi32>, %c: memref<5x6xi32>) {
  linalg.generic {indexing_maps = [#id, #transpose],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a : memref<6x5xi32>) outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %initial: i32):
    linalg.yield %x : i32
  }
  return
}

// AME-LABEL: func.func @advanced(
// AME-NOT: linalg.generic
// AME: ame.msll.ew
// AME: ame.msra.ew
// AME: ame.msrl.ew
// AME: ame.mmin.ew
// AME: ame.mmax.ew
// AME: ame.release
// AME: return
// ASM-LABEL: advanced:
// ASM: msll.ew
// ASM: msra.ew
// ASM: msrl.ew
// ASM: mmin.ew
// ASM: mmax.ew
// ASM: ame.release
func.func @advanced(%a: memref<5x6xi32>, %b: memref<5x6xi32>,
                    %c: memref<5x6xi32>, %capture: i32) {
  linalg.generic {indexing_maps = [#id, #id, #id],
                  iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<5x6xi32>, memref<5x6xi32>)
      outs(%c : memref<5x6xi32>) {
  ^bb0(%x: i32, %y: i32, %initial: i32):
    %mask = arith.constant 31 : i32
    %shift = arith.andi %y, %mask : i32
    %left = arith.shli %x, %shift : i32
    %arithmetic = arith.shrsi %left, %shift : i32
    %logical = arith.shrui %arithmetic, %shift : i32
    %min = arith.minui %logical, %x : i32
    %max = arith.maxui %min, %initial : i32
    %biased = arith.addi %max, %capture : i32
    %result = arith.minsi %biased, %x : i32
    linalg.yield %result : i32
  }
  return
}

// AME-LABEL: func.func @rank_one(
// AME-NOT: linalg.generic
// AME: ame.mbcast.m.x
// AME: ame.mmul.ew
// AME: ame.madd.ew
// AME: ame.release
// AME: return
func.func @rank_one(%a: memref<6xi32>, %c: memref<6xi32>, %scale: i32) {
  linalg.generic {indexing_maps = [affine_map<(i) -> (i)>,
                                  affine_map<(i) -> ()>,
                                  affine_map<(i) -> (i)>],
                  iterator_types = ["parallel"]}
      ins(%a, %scale : memref<6xi32>, i32) outs(%c : memref<6xi32>) {
  ^bb0(%x: i32, %factor: i32, %initial: i32):
    %bias = arith.constant -7 : i32
    %scaled = arith.muli %x, %factor : i32
    %result = arith.addi %scaled, %bias : i32
    linalg.yield %result : i32
  }
  return
}

func.func @main() -> i32 {
  %a_storage = memref.alloca() : memref<16x16xi32>
  %b_storage = memref.alloca() : memref<16x16xi32>
  %c_storage = memref.alloca() : memref<16x16xi32>
  %a = memref.subview %a_storage[2, 3] [5, 6] [2, 2] :
      memref<16x16xi32> to memref<5x6xi32, strided<[32, 2], offset: 35>>
  %b = memref.subview %b_storage[1, 2] [5, 6] [1, 2] :
      memref<16x16xi32> to memref<5x6xi32, strided<[16, 2], offset: 18>>
  %arithmetic = memref.subview %c_storage[3, 1] [5, 6] [1, 2] :
      memref<16x16xi32> to memref<5x6xi32, strided<[16, 2], offset: 49>>
  %ad = memref.cast %a : memref<5x6xi32, strided<[32, 2], offset: 35>>
      to memref<?x?xi32, strided<[?, ?], offset: ?>>
  %bd = memref.cast %b : memref<5x6xi32, strided<[16, 2], offset: 18>>
      to memref<?x?xi32, strided<[?, ?], offset: ?>>
  %cd = memref.cast %arithmetic : memref<5x6xi32, strided<[16, 2], offset: 49>>
      to memref<?x?xi32, strided<[?, ?], offset: ?>>
  %initial = memref.alloca() : memref<5x6xi32>
  %advanced = memref.alloca() : memref<5x6xi32>
  %advanced_a = memref.alloca() : memref<5x6xi32>
  %advanced_b = memref.alloca() : memref<5x6xi32>
  %rank_one = memref.alloca() : memref<6xi32>
  %transposed_input = memref.alloca() : memref<6x5xi32>
  %transposed = memref.alloca() : memref<5x6xi32>
  %permuted = memref.alloca() : memref<5x6xi32>
  %row = memref.alloca() : memref<6xi32>
  %column = memref.alloca() : memref<5xi32>
  %scalar = memref.alloca() : memref<i32>
  %broadcast = memref.alloca() : memref<5x6xi32>
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i5 = arith.constant 5 : index
  %i6 = arith.constant 6 : index
  %i16 = arith.constant 16 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %three = arith.constant 3 : i32
  %seven = arith.constant 7 : i32
  %eleven = arith.constant 11 : i32
  %scale = arith.constant 1073741825 : i32
  %scalar_value = arith.constant -19 : i32
  %mask = arith.constant 31 : i32
  %rank_bias = arith.constant -7 : i32
  %sentinel = arith.constant 99 : i32
  %false = arith.constant false
  scf.for %i = %i0 to %i16 step %i1 {
    scf.for %j = %i0 to %i16 step %i1 {
      memref.store %sentinel, %c_storage[%i, %j] : memref<16x16xi32>
    }
  }
  memref.store %scalar_value, %scalar[] : memref<i32>
  scf.for %i = %i0 to %i5 step %i1 {
    %iv = arith.index_cast %i : index to i32
    %column_product = arith.muli %iv, %eleven : i32
    %column_value = arith.subi %column_product, %seven : i32
    memref.store %column_value, %column[%i] : memref<5xi32>
    scf.for %j = %i0 to %i6 step %i1 {
      %jv = arith.index_cast %j : index to i32
      %difference = arith.subi %iv, %jv : i32
      %shifted = arith.subi %difference, %two : i32
      %av = arith.muli %shifted, %scale : i32
      %bv = arith.addi %difference, %three : i32
      %initialv = arith.addi %iv, %seven : i32
      memref.store %av, %a[%i, %j] :
          memref<5x6xi32, strided<[32, 2], offset: 35>>
      memref.store %bv, %b[%i, %j] :
          memref<5x6xi32, strided<[16, 2], offset: 18>>
      memref.store %initialv, %initial[%i, %j] : memref<5x6xi32>
      memref.store %initialv, %advanced[%i, %j] : memref<5x6xi32>
      memref.store %av, %advanced_a[%i, %j] : memref<5x6xi32>
      memref.store %bv, %advanced_b[%i, %j] : memref<5x6xi32>
      memref.store %initialv, %arithmetic[%i, %j] :
          memref<5x6xi32, strided<[16, 2], offset: 49>>
      memref.store %av, %transposed_input[%j, %i] : memref<6x5xi32>
    }
  }
  scf.for %j = %i0 to %i6 step %i1 {
    %jv = arith.index_cast %j : index to i32
    %row_product = arith.muli %jv, %three : i32
    %row_value = arith.subi %row_product, %eleven : i32
    memref.store %row_value, %row[%j] : memref<6xi32>
  }
  func.call @arithmetic(%ad, %bd, %cd) :
      (memref<?x?xi32, strided<[?, ?], offset: ?>>,
       memref<?x?xi32, strided<[?, ?], offset: ?>>,
       memref<?x?xi32, strided<[?, ?], offset: ?>>) -> ()
  func.call @transpose(%transposed_input, %transposed) :
      (memref<6x5xi32>, memref<5x6xi32>) -> ()
  func.call @permuted_output(%transposed_input, %permuted) :
      (memref<6x5xi32>, memref<5x6xi32>) -> ()
  func.call @broadcasts(%row, %column, %scalar, %broadcast) :
      (memref<6xi32>, memref<5xi32>, memref<i32>, memref<5x6xi32>) -> ()

  func.call @advanced(%advanced_a, %advanced_b, %advanced, %scalar_value) :
      (memref<5x6xi32>, memref<5x6xi32>, memref<5x6xi32>, i32) -> ()
  func.call @rank_one(%row, %rank_one, %scale) :
      (memref<6xi32>, memref<6xi32>, i32) -> ()

  // Empty shapes must not access any source or destination elements.
  %a0 = memref.subview %a[0, 0] [0, 6] [1, 1] :
      memref<5x6xi32, strided<[32, 2], offset: 35>> to
      memref<0x6xi32, strided<[32, 2], offset: 35>>
  %b0 = memref.subview %b[0, 0] [0, 6] [1, 1] :
      memref<5x6xi32, strided<[16, 2], offset: 18>> to
      memref<0x6xi32, strided<[16, 2], offset: 18>>
  %c0 = memref.subview %arithmetic[0, 0] [0, 6] [1, 1] :
      memref<5x6xi32, strided<[16, 2], offset: 49>> to
      memref<0x6xi32, strided<[16, 2], offset: 49>>
  %ad0 = memref.cast %a0 : memref<0x6xi32, strided<[32, 2], offset: 35>> to
      memref<?x?xi32, strided<[?, ?], offset: ?>>
  %bd0 = memref.cast %b0 : memref<0x6xi32, strided<[16, 2], offset: 18>> to
      memref<?x?xi32, strided<[?, ?], offset: ?>>
  %cd0 = memref.cast %c0 : memref<0x6xi32, strided<[16, 2], offset: 49>> to
      memref<?x?xi32, strided<[?, ?], offset: ?>>
  func.call @arithmetic(%ad0, %bd0, %cd0) :
      (memref<?x?xi32, strided<[?, ?], offset: ?>>,
       memref<?x?xi32, strided<[?, ?], offset: ?>>,
       memref<?x?xi32, strided<[?, ?], offset: ?>>) -> ()

  %bad = scf.for %i = %i0 to %i5 step %i1
      iter_args(%bad_rows = %false) -> (i1) {
    %bad_row = scf.for %j = %i0 to %i6 step %i1
        iter_args(%bad_columns = %bad_rows) -> (i1) {
      %x = memref.load %a[%i, %j] :
          memref<5x6xi32, strided<[32, 2], offset: 35>>
      %y = memref.load %b[%i, %j] :
          memref<5x6xi32, strided<[16, 2], offset: 18>>
      %initialv = memref.load %initial[%i, %j] : memref<5x6xi32>
      %sum = arith.addi %x, %y : i32
      %difference = arith.subi %sum, %initialv : i32
      %product = arith.muli %difference, %x : i32
      %and = arith.andi %product, %y : i32
      %or = arith.ori %and, %initialv : i32
      %xor = arith.xori %or, %product : i32
      %min = arith.minsi %xor, %x : i32
      %max = arith.maxsi %min, %y : i32
      %arithmetic_actual = memref.load %arithmetic[%i, %j] :
          memref<5x6xi32, strided<[16, 2], offset: 49>>
      %arithmetic_bad = arith.cmpi ne, %arithmetic_actual, %max : i32
      %transpose_actual = memref.load %transposed[%i, %j] : memref<5x6xi32>
      %transpose_bad = arith.cmpi ne, %transpose_actual, %x : i32
      %permuted_actual = memref.load %permuted[%i, %j] : memref<5x6xi32>
      %permuted_bad = arith.cmpi ne, %permuted_actual, %x : i32
      %either_transpose_bad = arith.ori %transpose_bad, %permuted_bad : i1
      %rowv = memref.load %row[%j] : memref<6xi32>
      %columnv = memref.load %column[%i] : memref<5xi32>
      %broadcast_sum = arith.addi %rowv, %columnv : i32
      %broadcast_expected = arith.subi %broadcast_sum, %scalar_value : i32
      %broadcast_actual = memref.load %broadcast[%i, %j] : memref<5x6xi32>
      %broadcast_bad = arith.cmpi ne, %broadcast_actual, %broadcast_expected : i32
      %arithmetic_transpose_bad = arith.ori %arithmetic_bad, %either_transpose_bad : i1
      %shift = arith.andi %y, %mask : i32
      %left = arith.shli %x, %shift : i32
      %arithmetic_shift = arith.shrsi %left, %shift : i32
      %logical = arith.shrui %arithmetic_shift, %shift : i32
      %unsigned_min = arith.minui %logical, %x : i32
      %unsigned_max = arith.maxui %unsigned_min, %initialv : i32
      %biased = arith.addi %unsigned_max, %scalar_value : i32
      %advanced_expected = arith.minsi %biased, %x : i32
      %advanced_actual = memref.load %advanced[%i, %j] : memref<5x6xi32>
      %advanced_bad = arith.cmpi ne, %advanced_actual, %advanced_expected : i32
      %rank_product = arith.muli %rowv, %scale : i32
      %rank_expected = arith.addi %rank_product, %rank_bias : i32
      %rank_actual = memref.load %rank_one[%j] : memref<6xi32>
      %rank_bad = arith.cmpi ne, %rank_actual, %rank_expected : i32
      %broadcast_rank_bad = arith.ori %broadcast_bad, %rank_bad : i1
      %any_bad = arith.ori %arithmetic_transpose_bad, %broadcast_rank_bad : i1
      %advanced_any_bad = arith.ori %any_bad, %advanced_bad : i1
      %next_bad = arith.ori %bad_columns, %advanced_any_bad : i1
      scf.yield %next_bad : i1
    }
    scf.yield %bad_row : i1
  }
  %outside = memref.load %c_storage[%i0, %i0] : memref<16x16xi32>
  %sentinel_changed = arith.cmpi ne, %outside, %sentinel : i32
  %failed = arith.ori %bad, %sentinel_changed : i1
  %status = arith.select %failed, %one, %zero : i32
  return %status : i32
}
