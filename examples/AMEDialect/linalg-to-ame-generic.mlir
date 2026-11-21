// RUN: buddy-opt %s --lower-linalg-to-ame='target=gem5' | FileCheck %s

// This generic region composes transpose, row/column broadcasts, a captured
// scalar and arithmetic. AME computes C = 2 * (transpose(A) + row) - column.
// Its 4x4 result is printed through gem5/print.c and independently checked.
// Exit: 0 = exact match; 1 = mismatch; 4 = print failure.

#id = affine_map<(m, n) -> (m, n)>
#transpose = affine_map<(m, n) -> (n, m)>
#row = affine_map<(m, n) -> (n)>
#column = affine_map<(m, n) -> (m)>

// CHECK-LABEL: func.func @main()
// CHECK-NOT: linalg.generic
// CHECK: ame.acquire
// CHECK: ame.mls.cm
// CHECK: ame.mrowbcast.ew.x
// CHECK: ame.mcolbcast.ew.x
// CHECK: ame.madd.ew
// CHECK: ame.mbcast.m.x
// CHECK: ame.mmul.ew
// CHECK: ame.msub.ew
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK: call @print_C

module {
  memref.global "private" constant @input_a : memref<4x4xi32> = dense<[
      [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]>
  memref.global "private" constant @input_row : memref<4xi32> = dense<[10, 20, 30, 40]>
  memref.global "private" constant @input_column : memref<4xi32> = dense<[1, 2, 3, 4]>
  func.func private @print_C(i32, i32, i32, i32) -> i32

  func.func @main() -> i32 {
    %a = memref.get_global @input_a : memref<4x4xi32>
    %row = memref.get_global @input_row : memref<4xi32>
    %column = memref.get_global @input_column : memref<4xi32>
    %c = memref.alloca() alignment = 64 : memref<4x4xi32>
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %scale = arith.constant 2 : i32
    %print_failure = arith.constant 4 : i32
    %false = arith.constant false

    linalg.generic {indexing_maps = [#transpose, #row, #column, #id],
                    iterator_types = ["parallel", "parallel"]}
        ins(%a, %row, %column : memref<4x4xi32>, memref<4xi32>, memref<4xi32>)
        outs(%c : memref<4x4xi32>) {
    ^bb0(%av: i32, %rv: i32, %cv: i32, %initial: i32):
      %sum = arith.addi %av, %rv : i32
      %scaled = arith.muli %sum, %scale : i32
      %result = arith.subi %scaled, %cv : i32
      linalg.yield %result : i32
    }

    %print_status = scf.for %i = %i0 to %i4 step %i1
        iter_args(%errors = %zero) -> (i32) {
      %v0 = memref.load %c[%i, %i0] : memref<4x4xi32>
      %v1 = memref.load %c[%i, %i1] : memref<4x4xi32>
      %v2 = memref.load %c[%i, %i2] : memref<4x4xi32>
      %v3 = memref.load %c[%i, %i3] : memref<4x4xi32>
      %printed = func.call @print_C(%v0, %v1, %v2, %v3) : (i32, i32, i32, i32) -> i32
      %next = arith.ori %errors, %printed : i32
      scf.yield %next : i32
    }

    %bad = scf.for %i = %i0 to %i4 step %i1
        iter_args(%bad_rows = %false) -> (i1) {
      %bad_row = scf.for %j = %i0 to %i4 step %i1
          iter_args(%bad_elements = %bad_rows) -> (i1) {
        %av = memref.load %a[%j, %i] : memref<4x4xi32>
        %rv = memref.load %row[%j] : memref<4xi32>
        %cv = memref.load %column[%i] : memref<4xi32>
        %sum = arith.addi %av, %rv : i32
        %scaled = arith.muli %sum, %scale : i32
        %expected = arith.subi %scaled, %cv : i32
        %actual = memref.load %c[%i, %j] : memref<4x4xi32>
        %mismatch = arith.cmpi ne, %actual, %expected : i32
        %next_bad = arith.ori %bad_elements, %mismatch : i1
        scf.yield %next_bad : i1
      }
      scf.yield %bad_row : i1
    }
    %numerical_status = arith.select %bad, %one, %zero : i32
    %print_failed = arith.cmpi ne, %print_status, %zero : i32
    %exit_code = arith.select %print_failed, %print_failure, %numerical_status : i32
    return %exit_code : i32
  }
}

// Expected C:
// 21 49 77 105
// 22 50 78 106
// 23 51 79 107
// 24 52 80 108
