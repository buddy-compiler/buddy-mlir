// RUN: buddy-opt %s --lower-ame --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts | buddy-translate -buddy-to-llvmir | buddy-llc -mtriple=riscv64 -mattr=+m,+xame,+xame-gem5 -target-abi=lp64 -verify-machineinstrs -filetype=obj -o %t.o

// Standalone 4 x 4 signed-i32 demo for the AME_gem5 profile.
// Like BOSCAMEDialect/bosc-mma.mlir, main initializes A/B/C and prints C.
// A contains 1 through 16 row by row; B is all ones. C/ACC start at zero.
// Initialization, ownership, descriptors, AME computation, and the independent
// scalar reference are all in MLIR. Only _start and print_C need runtime code.
// Signed i32 descriptor: 0x40000020 (1073741856), from ZTT v0.6.
// Exit: 0 = exact match; 1 = mismatch; 4 = print failure; 77 = unavailable.

!matrix = !llvm.target<"riscv.ztt.matrix", i32, 4, 4>
!accumulator = !llvm.target<"riscv.ztt.acc", i32, 4, 4>

module {
  memref.global "private" constant @input_a : memref<4x4xi32> = dense<[
      [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]>
  memref.global "private" constant @input_b : memref<4x4xi32> = dense<1>

  // Prints one row and returns zero on success. See gem5/print.c.
  func.func private @print_C(i32, i32, i32, i32) -> i32

  func.func @main() -> i32 {
    // Stack allocation avoids malloc/free and a target libc dependency.
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
    %unavailable = arith.constant 77 : i32
    %false = arith.constant false
    %zero64 = arith.constant 0 : i64
    %one64 = arith.constant 1 : i64
    %signed_i32 = arith.constant 1073741856 : i64

    // Initialize every element before the first matrix load.
    scf.for %row = %i0 to %i4 step %i1 {
      scf.for %col = %i0 to %i4 step %i1 {
        %av = memref.load %input_a[%row, %col] : memref<4x4xi32>
        %bv = memref.load %input_b[%row, %col] : memref<4x4xi32>
        memref.store %av, %a[%row, %col] : memref<4x4xi32>
        memref.store %bv, %b[%row, %col] : memref<4x4xi32>
        memref.store %zero, %c[%row, %col] : memref<4x4xi32>
      }
    }

    %a_address = memref.extract_aligned_pointer_as_index %a : memref<4x4xi32> -> index
    %b_address = memref.extract_aligned_pointer_as_index %b : memref<4x4xi32> -> index
    %c_address = memref.extract_aligned_pointer_as_index %c : memref<4x4xi32> -> index
    %a_integer = arith.index_cast %a_address : index to i64
    %b_integer = arith.index_cast %b_address : index to i64
    %c_integer = arith.index_cast %c_address : index to i64
    %a_ptr = llvm.inttoptr %a_integer : i64 to !llvm.ptr
    %b_ptr = llvm.inttoptr %b_integer : i64 to !llvm.ptr
    %c_ptr = llvm.inttoptr %c_integer : i64 to !llvm.ptr

    %owner = ame.acquire %zero64 : (i64) -> i64
    %owned_bit = arith.andi %owner, %one64 : i64
    %owned = arith.cmpi ne, %owned_bit, %zero64 : i64
    %exit_code = scf.if %owned -> (i32) {
      llvm.inline_asm has_side_effects "csrw amestatus, zero", "~{memory}" : () -> ()

      // Undefined passthroughs are only used for initial construction;
      // successful msettyp/asettyp establish descriptors and clear their span.
      %a_seed = llvm.mlir.undef : !matrix
      %b_seed = llvm.mlir.undef : !matrix
      %c_seed = llvm.mlir.undef : !matrix
      %acc_seed = llvm.mlir.undef : !accumulator
      %a_config = ame.msettyp %a_seed, %signed_i32 : (!matrix, i64) -> !matrix
      %b_config = ame.msettyp %b_seed, %signed_i32 : (!matrix, i64) -> !matrix
      %c_config = ame.msettyp %c_seed, %signed_i32 : (!matrix, i64) -> !matrix
      %acc_config = ame.asettyp %acc_seed, %signed_i32 : (!accumulator, i64) -> !accumulator
      %initial = ame.mzero.2d.acc %acc_config : (!accumulator) -> !accumulator
      %lhs = ame.mls.rm %a_config, %a_ptr : (!matrix, !llvm.ptr) -> !matrix
      %rhs = ame.mls.rm %b_config, %b_ptr : (!matrix, !llvm.ptr) -> !matrix
      %result = ame.mmulacc.2d %initial, %lhs, %rhs :
          (!accumulator, !matrix, !matrix) -> !accumulator
      %output = ame.mmov.m.a %c_config, %result : (!matrix, !accumulator) -> !matrix
      ame.mss.rm %output, %c_ptr : (!matrix, !llvm.ptr) -> ()

      %status = llvm.inline_asm has_side_effects "csrr $0, amestatus", "=r,~{memory}" : () -> i64
      %unsupported = arith.andi %status, %one64 : i64
      // No live AME value crosses release or an ordinary function call.
      ame.release : () -> ()
      %supported = arith.cmpi eq, %unsupported, %zero64 : i64
      %checked = scf.if %supported -> (i32) {
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

        // Scalar matrix multiplication is a reference only, never the source
        // of the printed C values. These small inputs cannot overflow i32.
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
        %checked_status = arith.select %print_failed, %print_failure, %numerical_status : i32
        scf.yield %checked_status : i32
      } else {
        scf.yield %unavailable : i32
      }
      scf.yield %checked : i32
    } else {
      scf.yield %unavailable : i32
    }
    func.return %exit_code : i32
  }
}

// Expected C = A * B (printed one row per line):
// 10 10 10 10
// 26 26 26 26
// 42 42 42 42
// 58 58 58 58
