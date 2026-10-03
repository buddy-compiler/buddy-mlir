// RUN: buddy-opt %s -lower-linalg-to-ime="target=k3" -lower-ime="target=k3" -expand-strided-metadata -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts | buddy-translate -buddy-to-llvmir | buddy-llc -mtriple=riscv64 -mcpu=spacemit-a100 -verify-machineinstrs -o - | FileCheck %s
// CHECK-DAG: smt.vmadot
// CHECK-DAG: smt.vfwmadot

// K3 hardware regression entry points, called by runtime_k3_dot_test.c.
// Build and run with: make k3-dot-test; scp k3-dot-test k3-005:/tmp/
func.func @loop_dot(%c: memref<1x1xi32>, %a: memref<1x1xi8>,
                   %b: memref<1x1xi8>) attributes {llvm.emit_c_interface} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 20000 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    ime.vmadot %c, %a, %b : memref<1x1xi32>, memref<1x1xi8>, memref<1x1xi8>
  }
  return
}
func.func @signed_dot(
    %c: memref<3x5xi32, strided<[?, ?], offset: ?>>,
    %a: memref<3x7xi8, strided<[?, ?], offset: ?>>,
    %b: memref<5x7xi8, strided<[?, ?], offset: ?>>)
    attributes {llvm.emit_c_interface} {
  ime.vmadot %c, %a, %b : memref<3x5xi32, strided<[?, ?], offset: ?>>,
    memref<3x7xi8, strided<[?, ?], offset: ?>>,
    memref<5x7xi8, strided<[?, ?], offset: ?>>
  return
}
func.func @unsigned_dot(
    %c: memref<3x5xi32, strided<[?, ?], offset: ?>>,
    %a: memref<3x7xui8, strided<[?, ?], offset: ?>>,
    %b: memref<5x7xui8, strided<[?, ?], offset: ?>>)
    attributes {llvm.emit_c_interface} {
  ime.vmadotu %c, %a, %b : memref<3x5xi32, strided<[?, ?], offset: ?>>,
    memref<3x7xui8, strided<[?, ?], offset: ?>>,
    memref<5x7xui8, strided<[?, ?], offset: ?>>
  return
}
func.func @signed_unsigned_dot(
    %c: memref<3x5xi32, strided<[?, ?], offset: ?>>,
    %a: memref<3x7xi8, strided<[?, ?], offset: ?>>,
    %b: memref<5x7xui8, strided<[?, ?], offset: ?>>)
    attributes {llvm.emit_c_interface} {
  ime.vmadotsu %c, %a, %b : memref<3x5xi32, strided<[?, ?], offset: ?>>,
    memref<3x7xi8, strided<[?, ?], offset: ?>>,
    memref<5x7xui8, strided<[?, ?], offset: ?>>
  return
}
func.func @unsigned_signed_dot(
    %c: memref<3x5xi32, strided<[?, ?], offset: ?>>,
    %a: memref<3x7xui8, strided<[?, ?], offset: ?>>,
    %b: memref<5x7xi8, strided<[?, ?], offset: ?>>)
    attributes {llvm.emit_c_interface} {
  ime.vmadotus %c, %a, %b : memref<3x5xi32, strided<[?, ?], offset: ?>>,
    memref<3x7xui8, strided<[?, ?], offset: ?>>,
    memref<5x7xi8, strided<[?, ?], offset: ?>>
  return
}
func.func @full_dot(%c: memref<8x8xi32>, %a: memref<8x16xi8>,
                   %b: memref<8x16xi8>) attributes {llvm.emit_c_interface} {
  ime.vmadot %c, %a, %b : memref<8x8xi32>, memref<8x16xi8>, memref<8x16xi8>
  return
}
func.func @fp16_dot(
    %c: memref<3x5xf16, strided<[?, ?], offset: ?>>,
    %a: memref<3x7xf16, strided<[?, ?], offset: ?>>,
    %b: memref<5x7xf16, strided<[?, ?], offset: ?>>)
    attributes {llvm.emit_c_interface} {
  ime.vfmadot %c, %a, %b : memref<3x5xf16, strided<[?, ?], offset: ?>>,
    memref<3x7xf16, strided<[?, ?], offset: ?>>,
    memref<5x7xf16, strided<[?, ?], offset: ?>>
  return
}

func.func @full_offset_dot(
    %c: memref<8x8xi32, strided<[8, 1], offset: ?>>,
    %a: memref<8x16xi8, strided<[16, 1], offset: ?>>,
    %b: memref<8x16xi8, strided<[16, 1], offset: ?>>)
    attributes {llvm.emit_c_interface} {
  ime.vmadot %c, %a, %b : memref<8x8xi32, strided<[8, 1], offset: ?>>,
    memref<8x16xi8, strided<[16, 1], offset: ?>>,
    memref<8x16xi8, strided<[16, 1], offset: ?>>
  return
}

// Multiple native tiles and boundaries along M, N and K, with nonzero C.
func.func @matmul(%a: memref<11x19xi8>, %b: memref<19x13xi8>,
                  %c: memref<11x13xi32>) attributes {llvm.emit_c_interface} {
  linalg.matmul ins(%a, %b : memref<11x19xi8>, memref<19x13xi8>)
                outs(%c : memref<11x13xi32>)
  return
}

func.func @generic_matmul(%a: memref<11x19xf16>, %b: memref<19x13xf16>,
                          %c: memref<11x13xf16>) attributes {llvm.emit_c_interface} {
  linalg.generic {indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                  affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
                  iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<11x19xf16>, memref<19x13xf16>)
      outs(%c : memref<11x13xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %p = arith.mulf %x, %y : f16
      %r = arith.addf %z, %p : f16
      linalg.yield %r : f16
  }
  return
}

func.func @batch_matmul(%a: memref<2x3x19xi8>, %b: memref<2x5x19xi8>,
                        %c: memref<2x3x5xi32>) attributes {llvm.emit_c_interface} {
  linalg.batch_matmul indexing_maps = [
      affine_map<(b,m,n,k)->(b,m,k)>, affine_map<(b,m,n,k)->(b,n,k)>,
      affine_map<(b,m,n,k)->(b,m,n)>]
      ins(%a, %b : memref<2x3x19xi8>, memref<2x5x19xi8>)
      outs(%c : memref<2x3x5xi32>)
  return
}
