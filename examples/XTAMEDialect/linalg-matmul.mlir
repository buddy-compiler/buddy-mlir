// RUN: buddy-opt %s -lower-linalg-to-xtame | FileCheck %s
// RUN: buddy-opt %s -lower-linalg-to-xtame -lower-xt-ame \
// RUN:   -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm -expand-strided-metadata -lower-affine \
// RUN:   -convert-arith-to-llvm -convert-math-to-llvm -convert-func-to-llvm \
// RUN:   -finalize-memref-to-llvm -reconcile-unrealized-casts | \
// RUN: buddy-translate -buddy-to-llvmir | \
// RUN: buddy-llc -filetype=asm -mtriple=riscv64-unknown-linux-gnu \
// RUN:   -mattr=+m,+a,+c,+d,+xtheadame -o - | FileCheck %s --check-prefix=ASM

// End-to-end matmul through the XTAME path.
//
//   C[8x8] = A[8x20] * B[20x8],  i8 * i8 -> i32
//
// The shape is chosen so that the lowering exercises every blocking rule:
// eight output rows fit one row tile, eight output columns become two
// four-column blocks, and K = 20 needs a chunk loop. `main` returns the number
// of mismatching elements, so the program is self-checking: a correct run
// exits with status 0 under the Xuantie QEMU model.
//
//   A[i, k] = 1 when k == i, else 0
//   B[k, l] = (k < 4 ? 1 : 2) + (l >= 4 ? 3 : 0)
//   C[i, l] = B[i, l]
//
// The reference is computed from the same expression, not from a second
// matmul, so an addressing error cannot cancel out.

module {
  func.func @main() -> i32 {
    %a = memref.alloca() : memref<8x20xi8>
    %b = memref.alloca() : memref<20x8xi8>
    %c = memref.alloca() : memref<8x8xi32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cm = arith.constant 8 : index
    %ck = arith.constant 20 : index
    %cn = arith.constant 8 : index
    %i8zero = arith.constant 0 : i8
    %i8one = arith.constant 1 : i8
    %i8two = arith.constant 2 : i8
    %i8three = arith.constant 3 : i8
    %i32zero = arith.constant 0 : i32
    %i32one = arith.constant 1 : i32
    %i32two = arith.constant 2 : i32
    %i32three = arith.constant 3 : i32

    scf.for %i = %c0 to %cm step %c1 {
      scf.for %k = %c0 to %ck step %c1 {
        %diagonal = arith.cmpi eq, %k, %i : index
        %value = arith.select %diagonal, %i8one, %i8zero : i8
        memref.store %value, %a[%i, %k] : memref<8x20xi8>
      }
    }

    scf.for %k = %c0 to %ck step %c1 {
      %lowRow = arith.cmpi ult, %k, %c4 : index
      %rowPart = arith.select %lowRow, %i8one, %i8two : i8
      scf.for %l = %c0 to %cn step %c1 {
        %highColumn = arith.cmpi uge, %l, %c4 : index
        %columnPart = arith.select %highColumn, %i8three, %i8zero : i8
        %value = arith.addi %rowPart, %columnPart : i8
        memref.store %value, %b[%k, %l] : memref<20x8xi8>
      }
    }

    scf.for %i = %c0 to %cm step %c1 {
      scf.for %l = %c0 to %cn step %c1 {
        memref.store %i32zero, %c[%i, %l] : memref<8x8xi32>
      }
    }

    linalg.matmul ins(%a, %b : memref<8x20xi8>, memref<20x8xi8>)
                  outs(%c : memref<8x8xi32>)

    %errors = scf.for %i = %c0 to %cm step %c1
        iter_args(%errorsI = %i32zero) -> (i32) {
      %lowRow = arith.cmpi ult, %i, %c4 : index
      %rowPart = arith.select %lowRow, %i32one, %i32two : i32
      %errorsL = scf.for %l = %c0 to %cn step %c1
          iter_args(%errorsLIn = %errorsI) -> (i32) {
        %actual = memref.load %c[%i, %l] : memref<8x8xi32>
        %highColumn = arith.cmpi uge, %l, %c4 : index
        %columnPart = arith.select %highColumn, %i32three, %i32zero : i32
        %expected = arith.addi %rowPart, %columnPart : i32
        %mismatch = arith.cmpi ne, %actual, %expected : i32
        %mismatch32 = arith.extui %mismatch : i1 to i32
        %errorsLOut = arith.addi %errorsLIn, %mismatch32 : i32
        scf.yield %errorsLOut : i32
      }
      scf.yield %errorsL : i32
    }

    func.return %errors : i32
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: linalg.matmul
// CHECK: xt_ame.th.mcfgni 4
// CHECK: xt_ame.th.mcfgmi 8
// CHECK: xt_ame.th.mlde32 0,
// CHECK: scf.for
// CHECK: xt_ame.th.mlde8 1,
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mmacc.w.b 3, 2, 1
// CHECK: xt_ame.th.mste32 0,
// CHECK: xt_ame.th.mste32 3,

// ASM-LABEL: main:
// ASM: th.mcfgmi{{[ \t]+}}8
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mlde32{{[ \t]+}}m3,
// ASM: th.mlde8{{[ \t]+}}m1,
// ASM: th.mldte8{{[ \t]+}}m2,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mmacc.w.b{{[ \t]+}}m3, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,
// ASM: th.mste32{{[ \t]+}}m3,
