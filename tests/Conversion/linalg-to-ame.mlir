// RUN: buddy-opt %s --lower-linalg-to-ame='target=gem5' | FileCheck %s --check-prefix=AME4
// RUN: buddy-opt %s --lower-linalg-to-ame | FileCheck %s --check-prefix=AME8

// The initial C is part of the computation, including a partial final tile.
func.func @static_tail(%a: memref<5x7xi32>, %b: memref<7x6xi32>,
                       %c: memref<5x6xi32>) {
  linalg.matmul ins(%a, %b : memref<5x7xi32>, memref<7x6xi32>)
                outs(%c : memref<5x6xi32>)
  return
}

// AME4-LABEL: func.func @static_tail(
// AME4: %[[ONE:.*]] = arith.constant 1 : i64
// AME4: %[[OWNER:.*]] = ame.acquire
// AME4: %[[GRANTED:.*]] = arith.andi %[[OWNER]], %[[ONE]] : i64
// AME4: %[[OWNED:.*]] = arith.cmpi ne, %[[GRANTED]], %{{.*}} : i64
// AME4: scf.if %[[OWNED]] {
// AME4: llvm.intr.stacksave
// AME4: ame.msettyp
// AME4: !llvm.target<"riscv.ztt.matrix", i32, 4, 4>
// AME4: ame.mls.rm
// AME4: ame.asettyp
// AME4: !llvm.target<"riscv.ztt.acc", i32, 4, 4>
// AME4: ame.mmov.a.m
// AME4: ame.mmulacc.2d
// AME4: ame.mmov.m.a
// AME4: ame.mss.rm
// AME4: llvm.intr.stackrestore
// AME4: ame.release
// AME4: } else {
// AME4-NOT: ame.
// AME4: arith.muli
// AME4: arith.addi
// AME4: memref.store
// AME4-NOT: ame.
// AME4: return
// AME8-LABEL: func.func @static_tail(
// AME8: ame.msettyp
// AME8: !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
// AME8: ame.asettyp
// AME8: !llvm.target<"riscv.ztt.acc", i32, 8, 8>
// AME8: ame.mmulacc.2d

// Floating point, packed narrow integers and tensor-form operations remain
// available for the caller's other lowering passes.
func.func @unsupported_f32(%a: memref<4x4xf32>, %b: memref<4x4xf32>,
                           %c: memref<4x4xf32>) {
  linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
                outs(%c : memref<4x4xf32>)
  return
}

// AME4-LABEL: func.func @unsupported_f32(
// AME4-NOT: ame.
// AME4: linalg.matmul
// AME4-NOT: ame.
// AME4: return

// Address spaces with a separate memory contract must not be treated as host
// pointers, including non-integer attributes used by downstream dialects.
func.func @unsupported_memory_space(%a: memref<4x4xi32, 1>,
                                    %b: memref<4x4xi32, 1>,
                                    %c: memref<4x4xi32, 1>) {
  linalg.matmul ins(%a, %b : memref<4x4xi32, 1>, memref<4x4xi32, 1>)
                outs(%c : memref<4x4xi32, 1>)
  return
}

// AME4-LABEL: func.func @unsupported_memory_space(
// AME4-NOT: ame.
// AME4: linalg.matmul
// AME4-NOT: ame.
// AME4: return

func.func @unsupported_symbolic_memory_space(
    %a: memref<4x4xi32, "device">, %b: memref<4x4xi32, "device">,
    %c: memref<4x4xi32, "device">) {
  linalg.matmul ins(%a, %b : memref<4x4xi32, "device">,
                             memref<4x4xi32, "device">)
                outs(%c : memref<4x4xi32, "device">)
  return
}

// AME4-LABEL: func.func @unsupported_symbolic_memory_space(
// AME4-NOT: ame.
// AME4: linalg.matmul
// AME4-NOT: ame.
// AME4: return

func.func @unsupported_transpose(%a: memref<7x5xi32>, %b: memref<7x6xi32>,
                                 %c: memref<5x6xi32>) {
  linalg.matmul indexing_maps = [
      affine_map<(m, n, k) -> (k, m)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>]
      ins(%a, %b : memref<7x5xi32>, memref<7x6xi32>)
      outs(%c : memref<5x6xi32>)
  return
}

// AME4-LABEL: func.func @unsupported_transpose(
// AME4-NOT: ame.
// AME4: linalg.matmul
// AME4-NOT: ame.
// AME4: return

func.func @unsupported_i8(%a: memref<4x4xi8>, %b: memref<4x4xi8>,
                          %c: memref<4x4xi32>) {
  linalg.matmul ins(%a, %b : memref<4x4xi8>, memref<4x4xi8>)
                outs(%c : memref<4x4xi32>)
  return
}

// AME4-LABEL: func.func @unsupported_i8(
// AME4-NOT: ame.
// AME4: linalg.matmul
// AME4-NOT: ame.
// AME4: return

func.func @tensor_form(%a: tensor<4x4xi32>, %b: tensor<4x4xi32>,
                       %c: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %result = linalg.matmul ins(%a, %b : tensor<4x4xi32>, tensor<4x4xi32>)
                          outs(%c : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %result : tensor<4x4xi32>
}

// AME4-LABEL: func.func @tensor_form(
// AME4-NOT: ame.
// AME4: linalg.matmul
// AME4-NOT: ame.
// AME4: return

// Dynamic dimensions must use runtime bounds, including M, N or K equal to 0.
func.func @dynamic(%a: memref<?x?xi32>, %b: memref<?x?xi32>,
                   %c: memref<?x?xi32>) {
  linalg.matmul ins(%a, %b : memref<?x?xi32>, memref<?x?xi32>)
                outs(%c : memref<?x?xi32>)
  return
}

// AME4-LABEL: func.func @dynamic(
// AME4: memref.dim
// AME4: memref.dim
// AME4: memref.dim
// AME4: ame.acquire
// AME4: scf.for
// AME4: memref.load
// AME4: ame.mmulacc.2d
// AME4: ame.mss.rm
// AME4: memref.store
// AME4: ame.release
// AME4: return

// Tile staging must access the logical memref view so offsets and arbitrary
// strides are respected; AME loads may only use the contiguous temporary tile.
func.func @strided(
    %a: memref<5x7xi32, strided<[?, ?], offset: ?>>,
    %b: memref<7x6xi32, strided<[?, ?], offset: ?>>,
    %c: memref<5x6xi32, strided<[?, ?], offset: ?>>) {
  linalg.matmul ins(%a, %b : memref<5x7xi32, strided<[?, ?], offset: ?>>,
                             memref<7x6xi32, strided<[?, ?], offset: ?>>)
                outs(%c : memref<5x6xi32, strided<[?, ?], offset: ?>>)
  return
}

// AME4-LABEL: func.func @strided(
// AME4: memref.alloca
// AME4: memref.load {{.*}} : memref<5x6xi32, strided<[?, ?], offset: ?>>
// AME4: ame.mmov.a.m
// AME4: memref.load {{.*}} : memref<5x7xi32, strided<[?, ?], offset: ?>>
// AME4: memref.load {{.*}} : memref<7x6xi32, strided<[?, ?], offset: ?>>
// AME4: ame.mmulacc.2d
// AME4: ame.mss.rm
// AME4: memref.store {{.*}} : memref<5x6xi32, strided<[?, ?], offset: ?>>
// AME4: ame.release
// AME4: return
