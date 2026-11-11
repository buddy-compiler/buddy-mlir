// RUN: buddy-opt %s -lower-linalg-to-xtame | FileCheck %s
// RUN: buddy-opt %s -lower-linalg-to-xtame -lower-xt-ame \
// RUN:   -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm -expand-strided-metadata -lower-affine \
// RUN:   -convert-arith-to-llvm -convert-math-to-llvm -convert-func-to-llvm \
// RUN:   -finalize-memref-to-llvm -reconcile-unrealized-casts | \
// RUN: buddy-translate -buddy-to-llvmir | \
// RUN: buddy-llc -filetype=asm -mtriple=riscv64-unknown-linux-gnu \
// RUN:   -mattr=+m,+a,+c,+d,+xtheadame -o - | FileCheck %s --check-prefix=ASM

module {
  func.func @matmul_i8(%a: memref<4x4xi8>,
                       %b: memref<4x4xi8>,
                       %c: memref<4x4xi32>) {
    linalg.matmul ins(%a, %b : memref<4x4xi8>, memref<4x4xi8>)
                  outs(%c : memref<4x4xi32>)
    return
  }

  func.func @matmul_i8_k8(%a: memref<4x8xi8>,
                          %b: memref<8x4xi8>,
                          %c: memref<4x4xi32>) {
    linalg.matmul ins(%a, %b : memref<4x8xi8>, memref<8x4xi8>)
                  outs(%c : memref<4x4xi32>)
    return
  }

  func.func @matmul_i8_k12(%a: memref<4x12xi8>,
                           %b: memref<12x4xi8>,
                           %c: memref<4x4xi32>) {
    linalg.matmul ins(%a, %b : memref<4x12xi8>, memref<12x4xi8>)
                  outs(%c : memref<4x4xi32>)
    return
  }

  func.func @matmul_i8_k20(%a: memref<4x20xi8>,
                           %b: memref<20x4xi8>,
                           %c: memref<4x4xi32>) {
    linalg.matmul ins(%a, %b : memref<4x20xi8>, memref<20x4xi8>)
                  outs(%c : memref<4x4xi32>)
    return
  }

  func.func @matmul_i8_m8(%a: memref<8x8xi8>,
                          %b: memref<8x4xi8>,
                          %c: memref<8x4xi32>) {
    linalg.matmul ins(%a, %b : memref<8x8xi8>, memref<8x4xi8>)
                  outs(%c : memref<8x4xi32>)
    return
  }

  func.func @matmul_i8_n8(%a: memref<4x8xi8>,
                          %b: memref<8x8xi8>,
                          %c: memref<4x8xi32>) {
    linalg.matmul ins(%a, %b : memref<4x8xi8>, memref<8x8xi8>)
                  outs(%c : memref<4x8xi32>)
    return
  }

  func.func @matmul_i8_k6(%a: memref<4x6xi8>,
                          %b: memref<6x4xi8>,
                          %c: memref<4x4xi32>) {
    linalg.matmul ins(%a, %b : memref<4x6xi8>, memref<6x4xi8>)
                  outs(%c : memref<4x4xi32>)
    return
  }

  func.func @matmul_i8_n5(%a: memref<4x4xi8>,
                          %b: memref<4x5xi8>,
                          %c: memref<4x5xi32>) {
    linalg.matmul ins(%a, %b : memref<4x4xi8>, memref<4x5xi8>)
                  outs(%c : memref<4x5xi32>)
    return
  }

  func.func @matmul_i8_m20(%a: memref<20x8xi8>,
                           %b: memref<8x4xi8>,
                           %c: memref<20x4xi32>) {
    linalg.matmul ins(%a, %b : memref<20x8xi8>, memref<8x4xi8>)
                  outs(%c : memref<20x4xi32>)
    return
  }

  // The smallest accepted shape and a shape where every extent is a remainder.
  func.func @matmul_i8_n1_k1(%a: memref<4x1xi8>,
                             %b: memref<1x1xi8>,
                             %c: memref<4x1xi32>) {
    linalg.matmul ins(%a, %b : memref<4x1xi8>, memref<1x1xi8>)
                  outs(%c : memref<4x1xi32>)
    return
  }

  func.func @matmul_i8_tails(%a: memref<5x3xi8>,
                             %b: memref<3x2xi8>,
                             %c: memref<5x2xi32>) {
    linalg.matmul ins(%a, %b : memref<5x3xi8>, memref<3x2xi8>)
                  outs(%c : memref<5x2xi32>)
    return
  }
}

// CHECK-LABEL: func.func @matmul_i8
// CHECK-NOT: linalg.matmul
// CHECK: xt_ame.th.mcfgni 4
// CHECK: xt_ame.th.mcfgmi 4
// CHECK: xt_ame.th.mcfgki 16
// Preserve the initial output tile because linalg.matmul accumulates into C.
// CHECK: xt_ame.th.mlde32 0,
// CHECK: xt_ame.th.mcfgki 4
// CHECK: xt_ame.th.mlde8 1,
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mcfgki 16
// CHECK: xt_ame.th.mste32 0,
// CHECK-NOT: linalg.matmul

// CHECK-LABEL: func.func @matmul_i8_k8
// CHECK-NOT: linalg.matmul
// The whole K extent fits into one register row, so a single
// multiply-accumulate replaces the reduction loop.
// CHECK-NOT: scf.for
// CHECK-NOT: memref.subview
// CHECK: xt_ame.th.mlde32 0,
// CHECK: xt_ame.th.mcfgki 8
// CHECK: %[[STRIDE_A:.*]] = arith.constant 8 : i64
// CHECK: xt_ame.th.mlde8 1, %[[STRIDE_A]],
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mste32 0,
// CHECK-NOT: linalg.matmul

// CHECK-LABEL: func.func @matmul_i8_k12
// CHECK-NOT: scf.for
// CHECK-NOT: memref.subview
// CHECK: xt_ame.th.mlde32 0,
// CHECK: xt_ame.th.mcfgki 12
// CHECK: %[[STRIDE_A12:.*]] = arith.constant 12 : i64
// CHECK: xt_ame.th.mlde8 1, %[[STRIDE_A12]],
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mcfgki 16
// CHECK: xt_ame.th.mste32 0,

// CHECK-LABEL: func.func @matmul_i8_k20
// K longer than one register row: a 16-element chunk loop first, then the 4
// remaining elements in a second loop, so K = 20 costs two chunks and not five.
// CHECK: xt_ame.th.mlde32 0,
// CHECK: xt_ame.th.mcfgki 16
// CHECK: scf.for %[[IV:.*]] = {{.*}} step {{.*}} {
// CHECK: memref.subview {{.*}}[0, %[[IV]]] [4, 16] [1, 1]
// CHECK: %[[STRIDE_A20:.*]] = arith.constant 20 : i64
// CHECK: xt_ame.th.mlde8 1, %[[STRIDE_A20]],
// CHECK: memref.subview {{.*}}[%[[IV]], 0] [16, 4] [1, 1]
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mcfgki 4
// CHECK: scf.for %[[IVT:.*]] = {{.*}} step {{.*}} {
// CHECK: memref.subview {{.*}}[0, %[[IVT]]] [4, 4] [1, 1]
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mste32 0,

// CHECK-LABEL: func.func @matmul_i8_m8
// The tile height follows M instead of being fixed at 4.
// CHECK-NOT: scf.for
// CHECK: xt_ame.th.mcfgni 4
// CHECK: xt_ame.th.mcfgmi 8
// CHECK: xt_ame.th.mcfgki 16
// CHECK: xt_ame.th.mlde32 0,
// CHECK: xt_ame.th.mcfgki 8
// CHECK: xt_ame.th.mlde8 1,
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mcfgki 16
// CHECK: xt_ame.th.mste32 0,

// CHECK-LABEL: func.func @matmul_i8_n8
// A is loaded once, before the column loop; the output is split into blocks
// of four columns.
// CHECK: xt_ame.th.mcfgki 8
// CHECK: %[[ASTRIDE:.*]] = arith.constant 8 : i64
// CHECK: xt_ame.th.mlde8 1, %[[ASTRIDE]],
// CHECK: scf.for %[[J:.*]] = {{.*}} to {{.*}} step {{.*}} {
// CHECK: memref.subview {{.*}}[0, %[[J]]] [4, 4] [1, 1] : memref<4x8xi32>
// CHECK: memref.subview {{.*}}[0, %[[J]]] [8, 4] [1, 1] : memref<8x8xi8>
// CHECK: xt_ame.th.mcfgki 16
// CHECK: %[[CSTRIDE:.*]] = arith.constant 32 : i64
// CHECK: xt_ame.th.mlde32 0, %[[CSTRIDE]],
// CHECK: xt_ame.th.mcfgki 8
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mcfgki 16
// CHECK: xt_ame.th.mste32 0,

// CHECK-LABEL: func.func @matmul_i8_k6
// A K extent of six elements still fits one register row.
// CHECK-NOT: scf.for
// CHECK: xt_ame.th.mcfgki 6
// CHECK: xt_ame.th.mlde8 1,
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1

// CHECK-LABEL: func.func @matmul_i8_n5
// Five output columns become a four-column block and a one-column tail.
// CHECK: xt_ame.th.mcfgki 4
// CHECK: xt_ame.th.mlde8 1,
// CHECK: memref.subview {{.*}}[0, 0] [4, 4] [1, 1] : memref<4x5xi32>
// CHECK: xt_ame.th.mcfgki 16
// CHECK: xt_ame.th.mlde32 0,
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mste32 0,
// CHECK: memref.subview {{.*}}[0, 4] [4, 1] [1, 1] : memref<4x5xi32>
// CHECK: xt_ame.th.mcfgki 4
// CHECK: xt_ame.th.mlde32 0,
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mste32 0,

// CHECK-LABEL: func.func @matmul_i8_m20
// Twenty output rows become a 16-row tile and a 4-row tile.
// CHECK: memref.subview {{.*}}[0, 0] [16, 8] [1, 1] : memref<20x8xi8>
// CHECK: xt_ame.th.mcfgmi 16
// CHECK: xt_ame.th.mlde8 1,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mste32 0,
// CHECK: memref.subview {{.*}}[16, 0] [4, 8] [1, 1] : memref<20x8xi8>
// CHECK: xt_ame.th.mcfgmi 4
// CHECK: xt_ame.th.mlde8 1,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mste32 0,

// CHECK-LABEL: func.func @matmul_i8_n1_k1
// Every extent is a tail. Each load and store is configured with the byte width
// of the subview it touches - one byte per row of A and B, four bytes per row
// of the one-column C block - so a narrow tile cannot access past its operand.
// CHECK: xt_ame.th.mcfgmi 4
// CHECK: xt_ame.th.mcfgki 4
// CHECK: xt_ame.th.mlde32 0,
// CHECK: xt_ame.th.mcfgki 1
// CHECK: xt_ame.th.mlde8 1,
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mcfgki 4
// CHECK: xt_ame.th.mste32 0,

// CHECK-LABEL: func.func @matmul_i8_tails
// One 5-row tile, a 3-element K tail and a 2-column N tail: the reduction tile
// is 3 bytes wide and the C block 8 bytes, matching the subviews exactly.
// CHECK: xt_ame.th.mcfgmi 5
// CHECK: xt_ame.th.mcfgki 8
// CHECK: xt_ame.th.mlde32 0,
// CHECK: xt_ame.th.mcfgki 3
// CHECK: xt_ame.th.mlde8 1,
// CHECK: xt_ame.th.mldte8 2,
// CHECK: xt_ame.th.mmacc.w.b 0, 2, 1
// CHECK: xt_ame.th.mste32 0,
// Spot-check the instruction selection reached through the full pipeline.
// ASM-LABEL: matmul_i8:
// ASM: th.mcfgni{{[ \t]+}}4
// ASM: th.mcfgmi{{[ \t]+}}4
// ASM: th.mcfgki{{[ \t]+}}16
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mcfgki{{[ \t]+}}4
// ASM: th.mlde8{{[ \t]+}}m1,
// ASM: th.mldte8{{[ \t]+}}m2,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mcfgki{{[ \t]+}}16
// ASM: th.mste32{{[ \t]+}}m0,

// ASM-LABEL: matmul_i8_k8:
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mcfgki{{[ \t]+}}8
// ASM: th.mlde8{{[ \t]+}}m1,
// ASM: th.mldte8{{[ \t]+}}m2,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,

// ASM-LABEL: matmul_i8_k12:
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mcfgki{{[ \t]+}}12
// ASM: th.mlde8{{[ \t]+}}m1,
// ASM: th.mldte8{{[ \t]+}}m2,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,

// ASM-LABEL: matmul_i8_k20:
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mcfgki{{[ \t]+}}4
// ASM: th.mlde8{{[ \t]+}}m1,
// ASM: th.mldte8{{[ \t]+}}m2,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,

// ASM-LABEL: matmul_i8_m8:
// ASM: th.mcfgni{{[ \t]+}}4
// ASM: th.mcfgmi{{[ \t]+}}8
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mcfgki{{[ \t]+}}8
// ASM: th.mlde8{{[ \t]+}}m1,
// ASM: th.mldte8{{[ \t]+}}m2,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,

// ASM-LABEL: matmul_i8_n8:
// ASM: th.mcfgmi{{[ \t]+}}4
// ASM: th.mcfgki{{[ \t]+}}8
// ASM: th.mlde8{{[ \t]+}}m1,
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mcfgki{{[ \t]+}}8
// ASM: th.mldte8{{[ \t]+}}m2,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,

// ASM-LABEL: matmul_i8_k6:
// ASM: th.mcfgmi{{[ \t]+}}4
// ASM: th.mcfgki{{[ \t]+}}16
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mcfgki{{[ \t]+}}6
// ASM: th.mlde8{{[ \t]+}}m1,
// ASM: th.mldte8{{[ \t]+}}m2,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,

// ASM-LABEL: matmul_i8_n5:
// ASM: th.mcfgki{{[ \t]+}}4
// ASM: th.mlde8{{[ \t]+}}m1,
// ASM: th.mcfgki{{[ \t]+}}16
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,
// ASM: th.mcfgki{{[ \t]+}}4
// ASM: th.mlde32{{[ \t]+}}m0,
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,

// ASM-LABEL: matmul_i8_m20:
// ASM: th.mcfgmi{{[ \t]+}}16
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,
// ASM: th.mcfgmi{{[ \t]+}}4
// ASM: th.mmacc.w.b{{[ \t]+}}m0, m2, m1
// ASM: th.mste32{{[ \t]+}}m0,
