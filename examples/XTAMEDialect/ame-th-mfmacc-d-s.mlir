// RUN: buddy-opt %s --lower-xt-ame | FileCheck %s

module {

  func.func private @print_C(f64, f64, f64, f64)

  func.func @main() -> i32 {

    %c_ptr = memref.alloc() : memref<4x4xf64>    // result matrix C (f64)
    %a_ptr = memref.alloc() : memref<4x4xf32>    // matrix A (f32)
    %b_ptr = memref.alloc() : memref<4x4xf32>    // matrix B (f32)

    // Stride calculations in bytes:
    // f32 is 4 bytes. 4 elements/row * 4 = 16 bytes per row
    // f64 is 8 bytes. 4 elements/row * 8 = 32 bytes per row
    %stride_a = arith.constant 16 : i64              // A's row stride
    %stride_b = arith.constant 16 : i64              // B's row stride
    %stride_c = arith.constant 32 : i64              // C's row stride

    //index
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

    //initialize A and B with float values
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32
    %v2 = arith.constant 2.0 : f32
    %v3 = arith.constant 3.0 : f32
    %v4 = arith.constant 4.0 : f32
    %v5 = arith.constant 5.0 : f32
    %v6 = arith.constant 6.0 : f32
    %v7 = arith.constant 7.0 : f32
    %v8 = arith.constant 8.0 : f32
    %v9 = arith.constant 9.0 : f32
    %v10 = arith.constant 10.0 : f32
    %v11 = arith.constant 11.0 : f32
    %v12 = arith.constant 12.0 : f32
    %v13 = arith.constant 13.0 : f32
    %v14 = arith.constant 14.0 : f32
    %v15 = arith.constant 15.0 : f32
    %v16 = arith.constant 16.0 : f32

    memref.store %v1, %a_ptr[%i0, %i0] : memref<4x4xf32>
    memref.store %v2, %a_ptr[%i0, %i1] : memref<4x4xf32>
    memref.store %v3, %a_ptr[%i0, %i2] : memref<4x4xf32>
    memref.store %v4, %a_ptr[%i0, %i3] : memref<4x4xf32>
    memref.store %v5, %a_ptr[%i1, %i0] : memref<4x4xf32>
    memref.store %v6, %a_ptr[%i1, %i1] : memref<4x4xf32>
    memref.store %v7, %a_ptr[%i1, %i2] : memref<4x4xf32>
    memref.store %v8, %a_ptr[%i1, %i3] : memref<4x4xf32>
    memref.store %v9, %a_ptr[%i2, %i0] : memref<4x4xf32>
    memref.store %v10, %a_ptr[%i2, %i1] : memref<4x4xf32>
    memref.store %v11, %a_ptr[%i2, %i2] : memref<4x4xf32>
    memref.store %v12, %a_ptr[%i2, %i3] : memref<4x4xf32>
    memref.store %v13, %a_ptr[%i3, %i0] : memref<4x4xf32>
    memref.store %v14, %a_ptr[%i3, %i1] : memref<4x4xf32>
    memref.store %v15, %a_ptr[%i3, %i2] : memref<4x4xf32>
    memref.store %v16, %a_ptr[%i3, %i3] : memref<4x4xf32>

    // Initialize B as an Identity matrix for simple testing
    memref.store %v1, %b_ptr[%i0, %i0] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i0, %i1] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i0, %i2] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i0, %i3] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i1, %i0] : memref<4x4xf32>
    memref.store %v1, %b_ptr[%i1, %i1] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i1, %i2] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i1, %i3] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i2, %i0] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i2, %i1] : memref<4x4xf32>
    memref.store %v1, %b_ptr[%i2, %i2] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i2, %i3] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i3, %i0] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i3, %i1] : memref<4x4xf32>
    memref.store %v0, %b_ptr[%i3, %i2] : memref<4x4xf32>
    memref.store %v1, %b_ptr[%i3, %i3] : memref<4x4xf32>

    // Step 1: Configure tile dimensions (4x4)
    xt_ame.th.mcfgmi 4          // mtilem = 4
    xt_ame.th.mcfgni 4          // mtilen = 4
    xt_ame.th.mcfgki 16         // mtilek = 16 (sizeK x 8 / MSEW)

    // Step 2: Zero the accumulation register (acc register 0)
    xt_ame.th.mzero 0

    // Step 3: Load float matrix A (32-bit elements) to tile register 1
    xt_ame.th.mlde32 1, %stride_a, %a_ptr: memref<4x4xf32>

    // Step 4: Load float transposed matrix B (32-bit elements) to tile register 2
    xt_ame.th.mldte32 2, %stride_b, %b_ptr: memref<4x4xf32>

    // Step 5: Execute float matrix multiply: acc0 = acc0 + tile1 x tile2 (f32 x f32 -> f64)
    xt_ame.th.mfmacc.d.s 0, 2, 1

    xt_ame.th.mcfgki 32         // f64 config adjustment if needed by your runtime

    // Step 6: Store f64 result from accumulator 0 to memory
    xt_ame.th.mste64 0, %stride_c, %c_ptr: memref<4x4xf64>

    //row 0
    %val_c00 = memref.load %c_ptr[%i0, %i0] : memref<4x4xf64>
    %val_c01 = memref.load %c_ptr[%i0, %i1] : memref<4x4xf64>
    %val_c02 = memref.load %c_ptr[%i0, %i2] : memref<4x4xf64>
    %val_c03 = memref.load %c_ptr[%i0, %i3] : memref<4x4xf64>
    call @print_C(%val_c00, %val_c01, %val_c02, %val_c03) : (f64, f64, f64, f64) -> ()

    //row 1
    %val_c10 = memref.load %c_ptr[%i1, %i0] : memref<4x4xf64>
    %val_c11 = memref.load %c_ptr[%i1, %i1] : memref<4x4xf64>
    %val_c12 = memref.load %c_ptr[%i1, %i2] : memref<4x4xf64>
    %val_c13 = memref.load %c_ptr[%i1, %i3] : memref<4x4xf64>
    call @print_C(%val_c10, %val_c11, %val_c12, %val_c13) : (f64, f64, f64, f64) -> ()

    //row 2
    %val_c20 = memref.load %c_ptr[%i2, %i0] : memref<4x4xf64>
    %val_c21 = memref.load %c_ptr[%i2, %i1] : memref<4x4xf64>
    %val_c22 = memref.load %c_ptr[%i2, %i2] : memref<4x4xf64>
    %val_c23 = memref.load %c_ptr[%i2, %i3] : memref<4x4xf64>
    call @print_C(%val_c20, %val_c21, %val_c22, %val_c23) : (f64, f64, f64, f64) -> ()

    //row 3
    %val_c30 = memref.load %c_ptr[%i3, %i0] : memref<4x4xf64>
    %val_c31 = memref.load %c_ptr[%i3, %i1] : memref<4x4xf64>
    %val_c32 = memref.load %c_ptr[%i3, %i2] : memref<4x4xf64>
    %val_c33 = memref.load %c_ptr[%i3, %i3] : memref<4x4xf64>
    call @print_C(%val_c30, %val_c31, %val_c32, %val_c33) : (f64, f64, f64, f64) -> ()

    memref.dealloc %c_ptr : memref<4x4xf64>
    memref.dealloc %a_ptr : memref<4x4xf32>
    memref.dealloc %b_ptr : memref<4x4xf32>

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
// CHECK: llvm.call @llvm.riscv.buddy.th.mlde32
// CHECK: llvm.call @llvm.riscv.buddy.th.mldte32
// CHECK: llvm.call @llvm.riscv.buddy.th.mfmacc.d.s
// CHECK: llvm.call @llvm.riscv.buddy.th.mste64
