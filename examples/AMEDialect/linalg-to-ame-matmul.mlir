// RUN: buddy-opt %s --lower-linalg-to-ame='target=gem5' | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK: ame.acquire
// CHECK: ame.msettyp
// CHECK: ame.mls.rm
// CHECK: ame.asettyp
// CHECK: ame.mmov.a.m
// CHECK: ame.mmulacc.2d
// CHECK: ame.mmov.m.a
// CHECK: ame.mss.rm
// CHECK: ame.release
// CHECK: call @print_C

// Linalg matmul -> AME (ZTT v0.6) -> LLVM IR -> RISC-V XAME.
// A contains 1 through 16 in row-major order; B is all ones and C starts at
// zero. The 4x4 tile matches the AME_gem5 profile. The initialized C result is
// printed and checked independently using scalar arithmetic.
// Exit: 0 = exact match; 1 = mismatch; 4 = print failure.

module {
  memref.global "private" constant @input_a : memref<4x4xi32> = dense<[
      [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]>
  memref.global "private" constant @input_b : memref<4x4xi32> = dense<1>

  // Uses the same gem5/print.c implementation as ame-matmul-demo.mlir.
  func.func private @print_C(i32, i32, i32, i32) -> i32

  func.func @main() -> i32 {
    %a = memref.alloca() alignment = 64 : memref<4x4xi32>
    %b = memref.alloca() alignment = 64 : memref<4x4xi32>
    %c = memref.alloca() alignment = 64 : memref<4x4xi32>
    %input_a = memref.get_global @input_a : memref<4x4xi32>
    %input_b = memref.get_global @input_b : memref<4x4xi32>
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %print_failure = arith.constant 4 : i32
    %false = arith.constant false

    scf.for %row = %i0 to %i4 step %i1 {
      scf.for %col = %i0 to %i4 step %i1 {
        %av = memref.load %input_a[%row, %col] : memref<4x4xi32>
        %bv = memref.load %input_b[%row, %col] : memref<4x4xi32>
        memref.store %av, %a[%row, %col] : memref<4x4xi32>
        memref.store %bv, %b[%row, %col] : memref<4x4xi32>
        memref.store %zero, %c[%row, %col] : memref<4x4xi32>
      }
    }

    // linalg.matmul accumulates A * B into the existing C values.
    linalg.matmul ins(%a, %b : memref<4x4xi32>, memref<4x4xi32>)
                  outs(%c : memref<4x4xi32>)

    %print_status = scf.for %row = %i0 to %i4 step %i1
        iter_args(%errors = %zero) -> (i32) {
      %v0 = memref.load %c[%row, %i0] : memref<4x4xi32>
      %v1 = memref.load %c[%row, %i1] : memref<4x4xi32>
      %v2 = memref.load %c[%row, %i2] : memref<4x4xi32>
      %v3 = memref.load %c[%row, %i3] : memref<4x4xi32>
      %printed = func.call @print_C(%v0, %v1, %v2, %v3) : (i32, i32, i32, i32) -> i32
      %next_errors = arith.ori %errors, %printed : i32
      scf.yield %next_errors : i32
    }

    // The scalar reference only checks C; it never provides the printed data.
    // These small input values cannot overflow i32.
    %bad = scf.for %row = %i0 to %i4 step %i1
        iter_args(%bad_rows = %false) -> (i1) {
      %bad_row = scf.for %col = %i0 to %i4 step %i1
          iter_args(%bad_elements = %bad_rows) -> (i1) {
        %expected = scf.for %k = %i0 to %i4 step %i1
            iter_args(%sum = %zero) -> (i32) {
          %av = memref.load %input_a[%row, %k] : memref<4x4xi32>
          %bv = memref.load %input_b[%k, %col] : memref<4x4xi32>
          %product = arith.muli %av, %bv : i32
          %next_sum = arith.addi %sum, %product : i32
          scf.yield %next_sum : i32
        }
        %actual = memref.load %c[%row, %col] : memref<4x4xi32>
        %different = arith.cmpi ne, %actual, %expected : i32
        %av = memref.load %a[%row, %col] : memref<4x4xi32>
        %bv = memref.load %b[%row, %col] : memref<4x4xi32>
        %original_a = memref.load %input_a[%row, %col] : memref<4x4xi32>
        %original_b = memref.load %input_b[%row, %col] : memref<4x4xi32>
        %changed_a = arith.cmpi ne, %av, %original_a : i32
        %changed_b = arith.cmpi ne, %bv, %original_b : i32
        %changed_input = arith.ori %changed_a, %changed_b : i1
        %bad_element = arith.ori %different, %changed_input : i1
        %next_bad = arith.ori %bad_elements, %bad_element : i1
        scf.yield %next_bad : i1
      }
      scf.yield %bad_row : i1
    }
    %numerical_status = arith.select %bad, %one, %zero : i32
    %print_failed = arith.cmpi ne, %print_status, %zero : i32
    %exit_code = arith.select %print_failed, %print_failure, %numerical_status : i32
    func.return %exit_code : i32
  }
}

// Expected C, printed one row per line:
// 10 10 10 10
// 26 26 26 26
// 42 42 42 42
// 58 58 58 58
