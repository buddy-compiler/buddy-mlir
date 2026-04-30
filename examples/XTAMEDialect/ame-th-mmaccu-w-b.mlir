// RUN: buddy-opt %s --lower-xt-ame | FileCheck %s

module {
  // Print the first 16 results for verification
  func.func private @print_C16(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

  func.func @main() -> i32 {

    // 1. Apply for large matrix memory
    %c_ptr = memref.alloc() : memref<64x64xi32>
    %a_ptr = memref.alloc() : memref<64x256xi8>
    %b_ptr = memref.alloc() : memref<256x64xi8>

    // 2. Define various constants
    %stride_a = arith.constant 256 : i64
    %stride_b = arith.constant 64 : i64
    %stride_c = arith.constant 256 : i64

    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %step1 = arith.constant 1 : index

    %v1 = arith.constant 1 : i8
    %v2 = arith.constant 2 : i8

    // ==========================================
    // 3. Data Initialization (Supplementary Complete Initialization Module)
    // ==========================================
    // Initialize A (64x256) with all ones
    scf.for %i = %c0 to %c64 step %step1 {
      scf.for %j = %c0 to %c256 step %step1 {
        memref.store %v1, %a_ptr[%i, %j] : memref<64x256xi8>
      }
    }

    // Initialize B (256x64) all to 2
    scf.for %i = %c0 to %c256 step %step1 {
      scf.for %j = %c0 to %c64 step %step1 {
        memref.store %v2, %b_ptr[%i, %j] : memref<256x64xi8>
      }
    }

    // ==========================================
    // 4. Matrix Command Configuration
    // ==========================================
    xt_ame.th.mcfgmi 16
    xt_ame.th.mcfgni 16
    xt_ame.th.mcfgki 64

    %step_m = arith.constant 16 : index
    %step_n = arith.constant 16 : index
    %step_k = arith.constant 64 : index

    // ==========================================
    // 5. Blocked Matrix Multiplication (Tiling)
    // ==========================================
    // Outer M loop (0, 16, 32, 48)
    scf.for %i = %c0 to %c64 step %step_m {

      // Mid-level N loop (0, 16, 32, 48)
      scf.for %j = %c0 to %c64 step %step_n {

        // Before calculating a new 16x16 block of C, the accumulator must be cleared
        xt_ame.th.mzero 0

        // The inner K loop accumulates (0, 64, 128, 192)
        scf.for %k = %c0 to %c256 step %step_k {

          // Extract subviews for the current tiles of A and B
          %sub_a = memref.subview %a_ptr[%i, %k] [16, 64] [1, 1]
                   : memref<64x256xi8> to memref<16x64xi8, strided<[256, 1], offset: ?>>

          %sub_b = memref.subview %b_ptr[%k, %j] [64, 16] [1, 1]
                   : memref<256x64xi8> to memref<64x16xi8, strided<[64, 1], offset: ?>>

          // Load and compute
          xt_ame.th.mlde8 1, %stride_a, %sub_a : memref<16x64xi8, strided<[256, 1], offset: ?>>
          xt_ame.th.mldte8 2, %stride_b, %sub_b : memref<64x16xi8, strided<[64, 1], offset: ?>>
          xt_ame.th.mmaccu.w.b 0, 2, 1
        }

        // Store back to C matrix memory
        %sub_c = memref.subview %c_ptr[%i, %j] [16, 16] [1, 1]
                 : memref<64x64xi32> to memref<16x16xi32, strided<[64, 1], offset: ?>>
        xt_ame.th.mste32 0, %stride_c, %sub_c : memref<16x16xi32, strided<[64, 1], offset: ?>>

      }
    }

    // ==========================================
    // 6. Print results verification (print the first 16 elements of the 0th row)
    // ==========================================
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    %i5 = arith.constant 5 : index
    %i6 = arith.constant 6 : index
    %i7 = arith.constant 7 : index
    %i8 = arith.constant 8 : index
    %i9 = arith.constant 9 : index
    %i10 = arith.constant 10 : index
    %i11 = arith.constant 11 : index
    %i12 = arith.constant 12 : index
    %i13 = arith.constant 13 : index
    %i14 = arith.constant 14 : index
    %i15 = arith.constant 15 : index

    %val0 = memref.load %c_ptr[%i0, %i0] : memref<64x64xi32>
    %val1 = memref.load %c_ptr[%i0, %i1] : memref<64x64xi32>
    %val2 = memref.load %c_ptr[%i0, %i2] : memref<64x64xi32>
    %val3 = memref.load %c_ptr[%i0, %i3] : memref<64x64xi32>
    %val4 = memref.load %c_ptr[%i0, %i4] : memref<64x64xi32>
    %val5 = memref.load %c_ptr[%i0, %i5] : memref<64x64xi32>
    %val6 = memref.load %c_ptr[%i0, %i6] : memref<64x64xi32>
    %val7 = memref.load %c_ptr[%i0, %i7] : memref<64x64xi32>
    %val8 = memref.load %c_ptr[%i0, %i8] : memref<64x64xi32>
    %val9 = memref.load %c_ptr[%i0, %i9] : memref<64x64xi32>
    %val10 = memref.load %c_ptr[%i0, %i10] : memref<64x64xi32>
    %val11 = memref.load %c_ptr[%i0, %i11] : memref<64x64xi32>
    %val12 = memref.load %c_ptr[%i0, %i12] : memref<64x64xi32>
    %val13 = memref.load %c_ptr[%i0, %i13] : memref<64x64xi32>
    %val14 = memref.load %c_ptr[%i0, %i14] : memref<64x64xi32>
    %val15 = memref.load %c_ptr[%i0, %i15] : memref<64x64xi32>

    call @print_C16(%val0, %val1, %val2, %val3, %val4, %val5, %val6, %val7, %val8, %val9, %val10, %val11, %val12, %val13, %val14, %val15)
         : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()

    // 7. Free memory
    memref.dealloc %c_ptr : memref<64x64xi32>
    memref.dealloc %a_ptr : memref<64x256xi8>
    memref.dealloc %b_ptr : memref<256x64xi8>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
// Expected lowering for tile-based operations:
// CHECK-LABEL: func.func @main
// CHECK: llvm.call @llvm.riscv.buddy.th.mcfgmi
// CHECK: llvm.call @llvm.riscv.buddy.th.mcfgni
// CHECK: llvm.call @llvm.riscv.buddy.th.mcfgki
// CHECK: llvm.call @llvm.riscv.buddy.th.mzero
// CHECK: llvm.call @llvm.riscv.buddy.th.mlde8
// CHECK: llvm.call @llvm.riscv.buddy.th.mldte8
// CHECK: llvm.call @llvm.riscv.buddy.th.mmaccu.w.b
// CHECK: llvm.call @llvm.riscv.buddy.th.mste32
