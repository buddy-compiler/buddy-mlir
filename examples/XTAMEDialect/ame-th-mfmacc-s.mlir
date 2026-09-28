// RUN: buddy-opt %s --lower-xt-ame | FileCheck %s
// RUN: buddy-opt %s \
// RUN:   --lower-xt-ame \
// RUN:   -convert-linalg-to-loops \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -expand-strided-metadata \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-math-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: buddy-translate -buddy-to-llvmir | \
// RUN: buddy-llc -filetype=obj -mtriple=riscv64-unknown-linux-gnu \
// RUN:   -mattr=+xtheadame -o %t.o && llvm-readobj --file-headers %t.o | FileCheck %s --check-prefix=OBJ

module {

  func.func private @print_C(f32, f32, f32, f32)

  func.func @main() -> i32 {

    %c_ptr = memref.alloc() : memref<4x4xf32>    // result matrix C (f32)
    %a_ptr = memref.alloc() : memref<4x8xf32>    // matrix A (f32)
    %b_ptr = memref.alloc() : memref<8x4xf32>    // matrix B (f32)

    // Stride calculations in bytes:
    // f32 is 4 bytes. 4 elements/row * 4 = 16 bytes per row
    %stride_a = arith.constant 32 : i64              // A's row stride
    %stride_b = arith.constant 16 : i64              // B's row stride
    %stride_c = arith.constant 16 : i64              // C's row stride

    //index
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    %i5 = arith.constant 5 : index
    %i6 = arith.constant 6 : index
    %i7 = arith.constant 7 : index

    //initialize A and B with float values
    %v1 = arith.constant 1.0 : f32
    %v2 = arith.constant 2.0 : f32

    // Row 0
    memref.store %v1, %a_ptr[%i0, %i0] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i0, %i1] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i0, %i2] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i0, %i3] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i0, %i4] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i0, %i5] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i0, %i6] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i0, %i7] : memref<4x8xf32>
    // Row 1
    memref.store %v1, %a_ptr[%i1, %i0] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i1, %i1] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i1, %i2] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i1, %i3] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i1, %i4] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i1, %i5] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i1, %i6] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i1, %i7] : memref<4x8xf32>
    // Row 2
    memref.store %v1, %a_ptr[%i2, %i0] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i2, %i1] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i2, %i2] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i2, %i3] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i2, %i4] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i2, %i5] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i2, %i6] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i2, %i7] : memref<4x8xf32>
    // Row 3
    memref.store %v1, %a_ptr[%i3, %i0] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i3, %i1] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i3, %i2] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i3, %i3] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i3, %i4] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i3, %i5] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i3, %i6] : memref<4x8xf32>
    memref.store %v1, %a_ptr[%i3, %i7] : memref<4x8xf32>

    // Initialize B as an Identity matrix for simple testing
    memref.store %v2, %b_ptr[%i0, %i0] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i0, %i1] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i0, %i2] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i0, %i3] : memref<8x4xf32>

    memref.store %v2, %b_ptr[%i1, %i0] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i1, %i1] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i1, %i2] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i1, %i3] : memref<8x4xf32>

    memref.store %v2, %b_ptr[%i2, %i0] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i2, %i1] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i2, %i2] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i2, %i3] : memref<8x4xf32>

    memref.store %v2, %b_ptr[%i3, %i0] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i3, %i1] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i3, %i2] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i3, %i3] : memref<8x4xf32>

    memref.store %v2, %b_ptr[%i4, %i0] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i4, %i1] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i4, %i2] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i4, %i3] : memref<8x4xf32>

    memref.store %v2, %b_ptr[%i5, %i0] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i5, %i1] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i5, %i2] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i5, %i3] : memref<8x4xf32>

    memref.store %v2, %b_ptr[%i6, %i0] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i6, %i1] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i6, %i2] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i6, %i3] : memref<8x4xf32>

    memref.store %v2, %b_ptr[%i7, %i0] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i7, %i1] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i7, %i2] : memref<8x4xf32>
    memref.store %v2, %b_ptr[%i7, %i3] : memref<8x4xf32>

    // Step 1: Configure tile dimensions (4x4)
    xt_ame.th.mcfgmi 4          // mtilem = 4
    xt_ame.th.mcfgni 4          // mtilen = 4
    xt_ame.th.mcfgki 32         // mtilek = 32

    // Step 2: Zero the accumulation register (acc register 0)
    %zero = xt_ame.th.mzero : vector<4x4xf32>

    // Step 3: Load float matrix A (32-bit elements) to tile register 1
    %lhs = xt_ame.th.mlde32 %stride_a, %a_ptr : memref<4x8xf32> -> vector<4x8xf32>

    // Step 4: Load float transposed matrix B (32-bit elements) to tile register 2
    %rhs = xt_ame.th.mldte32 %stride_b, %b_ptr: memref<8x4xf32> -> vector<8x4xf32>

    // Step 5: Execute float matrix multiply (f32 x f32 -> f32)
    %acc = xt_ame.th.mfmacc.s %zero, %rhs, %lhs : vector<4x4xf32>, vector<8x4xf32>, vector<4x8xf32> -> vector<4x4xf32>

    xt_ame.th.mcfgki 16

    // Step 6: Store f32 result from accumulator 0 to memory
    xt_ame.th.mste32 %acc, %stride_c, %c_ptr: vector<4x4xf32>, memref<4x4xf32>

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
    memref.dealloc %a_ptr : memref<4x8xf32>
    memref.dealloc %b_ptr : memref<8x4xf32>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}

// Expected lowering for tile-based operations:
// CHECK-LABEL: func.func @main
// CHECK: xt_ame.intr.th.mcfgmi
// CHECK: xt_ame.intr.th.mcfgni
// CHECK: xt_ame.intr.th.mcfgki
// CHECK: xt_ame.intr.th.mzero
// CHECK: xt_ame.intr.th.mlde32
// CHECK: xt_ame.intr.th.mldte32
// CHECK: xt_ame.intr.th.mfmacc.s
// CHECK: xt_ame.intr.th.mste32
// OBJ: Format: elf64-littleriscv
// OBJ: Machine: EM_RISCV
