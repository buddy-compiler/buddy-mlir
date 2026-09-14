// RUN: buddy-opt %s -lower-linalg-to-boscame
// RUN: buddy-opt %s \
// RUN:   -lower-linalg-to-boscame \
// RUN:   -lower-bosc-ame \
// RUN:   -convert-linalg-to-loops \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -expand-strided-metadata \
// RUN:   -lower-affine \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-math-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: buddy-translate -buddy-to-llvmir | \
// RUN: buddy-llc -filetype=obj -mtriple=riscv64 \
// RUN:   -mattr=+xboscame

// This file tests the lowering of linalg.matmul to boscame operations.
//
module {

  func.func private @print_C(i32, i32, i32, i32)

  func.func @matmul_i32_4x4x4(%A: memref<4x4xi32>, %B: memref<4x4xi32>, %C: memref<4x4xi32>) {
    linalg.matmul ins(%A, %B : memref<4x4xi32>, memref<4x4xi32>)
                  outs(%C : memref<4x4xi32>)
    return
  }

  func.func @main() -> i32 {

    %A = memref.alloc() : memref<4x4xi32>
    %B = memref.alloc() : memref<4x4xi32>
    %C = memref.alloc() : memref<4x4xi32>

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32

    linalg.fill ins(%c1_i32 : i32) outs(%A : memref<4x4xi32>)
    linalg.fill ins(%c2_i32 : i32) outs(%B : memref<4x4xi32>)
    linalg.fill ins(%c0_i32 : i32) outs(%C : memref<4x4xi32>)

    call @matmul_i32_4x4x4(%A, %B, %C) : (memref<4x4xi32>, memref<4x4xi32>, memref<4x4xi32>) -> ()

    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

    %val_c00 = memref.load %C[%i0, %i0] : memref<4x4xi32>
    %val_c01 = memref.load %C[%i0, %i1] : memref<4x4xi32>
    %val_c02 = memref.load %C[%i0, %i2] : memref<4x4xi32>
    %val_c03 = memref.load %C[%i0, %i3] : memref<4x4xi32>
    call @print_C(%val_c00, %val_c01, %val_c02, %val_c03) : (i32, i32, i32, i32) -> ()

    %val_c10 = memref.load %C[%i1, %i0] : memref<4x4xi32>
    %val_c11 = memref.load %C[%i1, %i1] : memref<4x4xi32>
    %val_c12 = memref.load %C[%i1, %i2] : memref<4x4xi32>
    %val_c13 = memref.load %C[%i1, %i3] : memref<4x4xi32>
    call @print_C(%val_c10, %val_c11, %val_c12, %val_c13) : (i32, i32, i32, i32) -> ()

    %val_c20 = memref.load %C[%i2, %i0] : memref<4x4xi32>
    %val_c21 = memref.load %C[%i2, %i1] : memref<4x4xi32>
    %val_c22 = memref.load %C[%i2, %i2] : memref<4x4xi32>
    %val_c23 = memref.load %C[%i2, %i3] : memref<4x4xi32>
    call @print_C(%val_c20, %val_c21, %val_c22, %val_c23) : (i32, i32, i32, i32) -> ()

    %val_c30 = memref.load %C[%i3, %i0] : memref<4x4xi32>
    %val_c31 = memref.load %C[%i3, %i1] : memref<4x4xi32>
    %val_c32 = memref.load %C[%i3, %i2] : memref<4x4xi32>
    %val_c33 = memref.load %C[%i3, %i3] : memref<4x4xi32>
    call @print_C(%val_c30, %val_c31, %val_c32, %val_c33) : (i32, i32, i32, i32) -> ()

    memref.dealloc %A : memref<4x4xi32>
    memref.dealloc %B : memref<4x4xi32>
    memref.dealloc %C : memref<4x4xi32>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
