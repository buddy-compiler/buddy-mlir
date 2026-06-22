// RUN: buddy-opt %s --lower-xt-ame | FileCheck %s

module {

  func.func private @print_C(i32, i32, i32, i32)

  func.func @main() -> i32 {

    %c_ptr = memref.alloc() : memref<4x4xi32>    // result matrix C
    %a_ptr = memref.alloc() : memref<4x4xi8>     // matrix A
    %b_ptr = memref.alloc() : memref<4x4xi8>     // matrix B

    %stride_a = arith.constant 4 : i64               // A's row stride (4 bytes per row)
    %stride_b = arith.constant 4 : i64               // B's row stride (4 bytes per row)
    %stride_c = arith.constant 16 : i64              // C's row stride (16 bytes per row, since each element is 4 bytes)

    //index
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

    //initialize A and B with some values (for testing)
    %v0 = arith.constant 0 : i8
    %v1 = arith.constant 1 : i8
    %v2 = arith.constant 2 : i8
    %v3 = arith.constant 3 : i8
    %v4 = arith.constant 4 : i8
    %v5 = arith.constant 5 : i8
    %v6 = arith.constant 6 : i8
    %v7 = arith.constant 7 : i8
    %v8 = arith.constant 8 : i8
    %v9 = arith.constant 9 : i8
    %v10 = arith.constant 10 : i8
    %v11 = arith.constant 11 : i8
    %v12 = arith.constant 12 : i8
    %v13 = arith.constant 13 : i8
    %v14 = arith.constant 14 : i8
    %v15 = arith.constant 15 : i8
    %v16 = arith.constant 16 : i8

    memref.store %v1, %a_ptr[%i0, %i0] : memref<4x4xi8>
    memref.store %v2, %a_ptr[%i0, %i1] : memref<4x4xi8>
    memref.store %v3, %a_ptr[%i0, %i2] : memref<4x4xi8>
    memref.store %v4, %a_ptr[%i0, %i3] : memref<4x4xi8>
    memref.store %v5, %a_ptr[%i1, %i0] : memref<4x4xi8>
    memref.store %v6, %a_ptr[%i1, %i1] : memref<4x4xi8>
    memref.store %v7, %a_ptr[%i1, %i2] : memref<4x4xi8>
    memref.store %v8, %a_ptr[%i1, %i3] : memref<4x4xi8>
    memref.store %v9, %a_ptr[%i2, %i0] : memref<4x4xi8>
    memref.store %v10, %a_ptr[%i2, %i1] : memref<4x4xi8>
    memref.store %v11, %a_ptr[%i2, %i2] : memref<4x4xi8>
    memref.store %v12, %a_ptr[%i2, %i3] : memref<4x4xi8>
    memref.store %v13, %a_ptr[%i3, %i0] : memref<4x4xi8>
    memref.store %v14, %a_ptr[%i3, %i1] : memref<4x4xi8>
    memref.store %v15, %a_ptr[%i3, %i2] : memref<4x4xi8>
    memref.store %v16, %a_ptr[%i3, %i3] : memref<4x4xi8>

    memref.store %v1, %b_ptr[%i0, %i0] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i0, %i1] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i0, %i2] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i0, %i3] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i1, %i0] : memref<4x4xi8>
    memref.store %v1, %b_ptr[%i1, %i1] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i1, %i2] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i1, %i3] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i2, %i0] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i2, %i1] : memref<4x4xi8>
    memref.store %v1, %b_ptr[%i2, %i2] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i2, %i3] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i3, %i0] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i3, %i1] : memref<4x4xi8>
    memref.store %v0, %b_ptr[%i3, %i2] : memref<4x4xi8>
    memref.store %v1, %b_ptr[%i3, %i3] : memref<4x4xi8>

    // Step 1: Configure tile dimensions
    // For a simple 4x4 tile operation
    xt_ame.th.mcfgmi 4          // mtilem = 4 (rows of C and A)
    xt_ame.th.mcfgni 4          // mtilen = 4 (cols of C and B)
    xt_ame.th.mcfgki 4          // mtilek = 4 (cols of A, rows of B)

    // Step 2: Zero the accumulation register (acc register 0)
    xt_ame.th.mzero 0

    // Step 3: Load matrix A to tile register 0 (shape: mtilem x mtilek = 4x4)
    xt_ame.th.mlde8 1, %stride_a, %a_ptr: memref<4x4xi8>

    // Step 4: Load transposed matrix B to tile register 1 (shape: mtilen x mtilek = 4x4)
    xt_ame.th.mldte8 2, %stride_b, %b_ptr: memref<4x4xi8>

    // Step 5: Execute matrix multiply: acc0 = acc0 + tile0 x tile1
    xt_ame.th.mmacc.w.b 0, 2, 1

    xt_ame.th.mcfgki 16         // mtilek = 16
    // Step 6: Store result from accumulator 0 to memory
    xt_ame.th.mste32 0, %stride_c, %c_ptr: memref<4x4xi32>

    //row 0
    %val_c00 = memref.load %c_ptr[%i0, %i0] : memref<4x4xi32>
    %val_c01 = memref.load %c_ptr[%i0, %i1] : memref<4x4xi32>
    %val_c02 = memref.load %c_ptr[%i0, %i2] : memref<4x4xi32>
    %val_c03 = memref.load %c_ptr[%i0, %i3] : memref<4x4xi32>
    call @print_C(%val_c00, %val_c01, %val_c02, %val_c03) : (i32, i32, i32, i32) -> ()

    //row 1
    %val_c10 = memref.load %c_ptr[%i1, %i0] : memref<4x4xi32>
    %val_c11 = memref.load %c_ptr[%i1, %i1] : memref<4x4xi32>
    %val_c12 = memref.load %c_ptr[%i1, %i2] : memref<4x4xi32>
    %val_c13 = memref.load %c_ptr[%i1, %i3] : memref<4x4xi32>
    call @print_C(%val_c10, %val_c11, %val_c12, %val_c13) : (i32, i32, i32, i32) -> ()

    //row 2
    %val_c20 = memref.load %c_ptr[%i2, %i0] : memref<4x4xi32>
    %val_c21 = memref.load %c_ptr[%i2, %i1] : memref<4x4xi32>
    %val_c22 = memref.load %c_ptr[%i2, %i2] : memref<4x4xi32>
    %val_c23 = memref.load %c_ptr[%i2, %i3] : memref<4x4xi32>
    call @print_C(%val_c20, %val_c21, %val_c22, %val_c23) : (i32, i32, i32, i32) -> ()

    //row 3
    %val_c30 = memref.load %c_ptr[%i3, %i0] : memref<4x4xi32>
    %val_c31 = memref.load %c_ptr[%i3, %i1] : memref<4x4xi32>
    %val_c32 = memref.load %c_ptr[%i3, %i2] : memref<4x4xi32>
    %val_c33 = memref.load %c_ptr[%i3, %i3] : memref<4x4xi32>
    call @print_C(%val_c30, %val_c31, %val_c32, %val_c33) : (i32, i32, i32, i32) -> ()

    memref.dealloc %c_ptr : memref<4x4xi32>
    memref.dealloc %a_ptr : memref<4x4xi8>
    memref.dealloc %b_ptr : memref<4x4xi8>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
  // For now, we only test the tile-level operations which map directly
  // to LLVM intrinsics.
}

// Expected lowering for tile-based operations:
// CHECK-LABEL: func.func @main
// CHECK: llvm.call @llvm.riscv.buddy.th.mcfgmi
// CHECK: llvm.call @llvm.riscv.buddy.th.mcfgni
// CHECK: llvm.call @llvm.riscv.buddy.th.mcfgki
// CHECK: llvm.call @llvm.riscv.buddy.th.mzero
// CHECK: llvm.call @llvm.riscv.buddy.th.mlde8
// CHECK: llvm.call @llvm.riscv.buddy.th.mldte8
// CHECK: llvm.call @llvm.riscv.buddy.th.mmacc.w.b
// CHECK: llvm.call @llvm.riscv.buddy.th.mste32
