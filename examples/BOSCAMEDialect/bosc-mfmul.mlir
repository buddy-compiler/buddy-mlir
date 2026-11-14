// RUN: buddy-opt %s --lower-bosc-ame
//
// ===========================================================================
// Complete Matrix Multiplication Demo using RISC-V Matrix Extension (BOSC MFMUL)
// ===========================================================================
//
// This demo shows the complete flow of matrix multiplication:
// 1. Configure element type (msettypei)
// 2. Configure tile dimensions (msettilemi, msettileni, msettileki)
// 3. Load the accumulation register
// 4. Zero accumulator (mfsub)
// 5. Load matrix tiles (mlce32.m, mlce32.m)
// 6. Execute matrix multiply (mfmul.f.mm)
// 7. Store result (msce32.m)
//
// Matrix dimensions: C[M×N] = A[M×K] × B[K×N]
// Tile dimensions are configured via msettilem/msettilen/msettilek
//
// ===========================================================================

module {

  func.func private @print_C(f32, f32, f32, f32)
  // Demo: float32 tile-based matrix multiplication
  // Uses tile register operations (hardware-level abstraction)
  func.func @main() -> i32 {

    %c_ptr = memref.alloc() : memref<4x4xf32>     // result matrix C
    %a_ptr = memref.alloc() : memref<4x4xf32>     // matrix A
    %b_ptr = memref.alloc() : memref<4x4xf32>     // matrix B

    %stride_a = arith.constant 16 : i64           // row stride for A
    %stride_b = arith.constant 16 : i64           // row stride for B
    %stride_c = arith.constant 16 : i64           // row stride for C

    //index
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

    //initialize A and B with some values (for testing)
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

    // Step 1: Configure tile dimensions
    // For a simple 4x4 tile operation
    %rd = bosc_ame.msettypei 32 : i64            // msettype(e32)

    %rd_m = bosc_ame.msettilemi 4 : i64          // mtilem = 4 (rows of C and A)
    %rd_n = bosc_ame.msettileni 4 : i64          // mtilen = 4 (cols of C and B)
    %rd_k = bosc_ame.msettileki 4 : i64          // mtilek = 4 (cols of A, rows of B)

    // Step 2: Load the accumulation register (tile register 0)
    %zero = bosc_ame.mlce32.m %c_ptr, %stride_c : memref<4x4xf32> -> vector<4x4xf32>

    // Step 3: Zero the accumulation register (tile register 0)
    %md = bosc_ame.mfsub.f.mm %zero, %zero : vector<4x4xf32>, vector<4x4xf32> -> vector<4x4xf32>

    // Step 4: Load matrix A to tile register 0 (shape: mtilem x mtilek = 4x4)
    %lhs = bosc_ame.mlae32.m %a_ptr, %stride_a : memref<4x4xf32> -> vector<4x4xf32>

    // Step 5: Load matrix B to tile register 1 (shape: mtilek x mtilen = 4x4)
    %rhs = bosc_ame.mlbe32.m %b_ptr, %stride_b : memref<4x4xf32> -> vector<4x4xf32>

    // Step 6: Execute element-wise matrix multiplication.
    %acc = bosc_ame.mfmul.f.mm %lhs, %rhs : vector<4x4xf32>, vector<4x4xf32> -> vector<4x4xf32>

    // Step 7: Store result from accumulator 0 to memory
    bosc_ame.msce32.m %acc, %c_ptr, %stride_c : vector<4x4xf32>, memref<4x4xf32>

    //row 0
    %val_c00 = memref.load %c_ptr[%i0, %i0] : memref<4x4xf32>
    %val_c01 = memref.load %c_ptr[%i0, %i1] : memref<4x4xf32>
    %val_c02 = memref.load %c_ptr[%i0, %i2] : memref<4x4xf32>
    %val_c03 = memref.load %c_ptr[%i0, %i3] : memref<4x4xf32>
    call @print_C(%val_c00, %val_c01, %val_c02, %val_c03) : (f32, f32, f32, f32) -> ()

    //row 1
    %val_c10 = memref.load %c_ptr[%i1, %i0] : memref<4x4xf32>
    %val_c11 = memref.load %c_ptr[%i1, %i1] : memref<4x4xf32>
    %val_c12 = memref.load %c_ptr[%i1, %i2] : memref<4x4xf32>
    %val_c13 = memref.load %c_ptr[%i1, %i3] : memref<4x4xf32>
    call @print_C(%val_c10, %val_c11, %val_c12, %val_c13) : (f32, f32, f32, f32) -> ()

    //row 2
    %val_c20 = memref.load %c_ptr[%i2, %i0] : memref<4x4xf32>
    %val_c21 = memref.load %c_ptr[%i2, %i1] : memref<4x4xf32>
    %val_c22 = memref.load %c_ptr[%i2, %i2] : memref<4x4xf32>
    %val_c23 = memref.load %c_ptr[%i2, %i3] : memref<4x4xf32>
    call @print_C(%val_c20, %val_c21, %val_c22, %val_c23) : (f32, f32, f32, f32) -> ()

    //row 3
    %val_c30 = memref.load %c_ptr[%i3, %i0] : memref<4x4xf32>
    %val_c31 = memref.load %c_ptr[%i3, %i1] : memref<4x4xf32>
    %val_c32 = memref.load %c_ptr[%i3, %i2] : memref<4x4xf32>
    %val_c33 = memref.load %c_ptr[%i3, %i3] : memref<4x4xf32>
    call @print_C(%val_c30, %val_c31, %val_c32, %val_c33) : (f32, f32, f32, f32) -> ()

    memref.dealloc %c_ptr : memref<4x4xf32>
    memref.dealloc %a_ptr : memref<4x4xf32>
    memref.dealloc %b_ptr : memref<4x4xf32>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }

  // NOTE: High-level mma.w.mm operation (memref abstraction) requires
  // additional lowering pass to convert memref to tile operations.
  // For now, we only test the tile-level operations which map directly
  // to LLVM intrinsics.
}
