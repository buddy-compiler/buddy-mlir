// RUN: buddy-opt %s --lower-bosc-ame | FileCheck %s --check-prefix=LOWER
// RUN: buddy-opt %s --lower-bosc-ame-tile | FileCheck %s --check-prefix=LOWER
// RUN: buddy-opt %s --lower-bosc-ame | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=IR

// Lowering of the five tile operations to ordered LLVM target intrinsics.
//
// The problem is deliberately non-square (M=8, K=16, N=32) and the B operand is
// pitched (K = 16 contiguous inside physical rows that are 24 bytes apart), so
//
//   * a swapped M/N/K shows up in the configuration arguments,
//   * using `strides[0]` for B instead of the physical row pitch shows up as a
//     byte stride of 1 instead of 24,
//   * the A row stride (16) differs from the B row pitch (24), which an
//     M = N = K = 64 problem could never distinguish.
//
// The tile operations lower directly to ordered `llvm.call_intrinsic` with
// LLVM pointers; `--lower-bosc-ame-tile` produces the same IR.
//
// The LLVM IR is where the values are checkable: after translation the
// configuration and the strides are literal operands of the calls,
// so the `IR` checks below verify the *values* and their order, while the
// `LOWER` checks only verify the structure of the LLVM-dialect output.

module attributes {bosc_ame.target = "nr-fpga"} {
  // LOWER-LABEL: llvm.func @non_square
  func.func @non_square() -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32

    %a_buf = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<1024xi8, #bosc_ame.spm<0>>
    %b_buf = bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64} : memref<1024xi8, #bosc_ame.spm<1>>
    %c_buf = bosc_ame.spm.view {bank = 0 : i32, offset = 2048 : i64} : memref<1024xi8, #bosc_ame.spm<0>>

    // A[8, 16] row-major, B logical [16, 32] over a physical [32, 16] buffer
    // with 24-byte rows, C[8, 32] contiguous i32.
    %a = memref.reinterpret_cast %a_buf to offset: [0], sizes: [8, 16], strides: [16, 1] : memref<1024xi8, #bosc_ame.spm<0>> to memref<8x16xi8, strided<[16, 1], offset: 0>, #bosc_ame.spm<0>>
    %b = memref.reinterpret_cast %b_buf to offset: [0], sizes: [16, 32], strides: [1, 24] : memref<1024xi8, #bosc_ame.spm<1>> to memref<16x32xi8, strided<[1, 24], offset: 0>, #bosc_ame.spm<1>>
    %c = memref.view %c_buf[%c0][] : memref<1024xi8, #bosc_ame.spm<0>> to memref<8x32xi32, #bosc_ame.spm<0>>

    // The accumulator load programs M = 8, N = 32 and the i32 accumulator
    // phase (0x10042 = 65602) before reading rows of 4 * 32 = 128 bytes from
    // the accumulator. The check pins the complete load configuration.
    // IR: call <vscale x 32 x i32> @llvm.riscv.bosc.ordered.load.acc.nxv32i32(ptr {{.*}}, i64 128, i64 8, i64 32, i64 65602)
    %acc0 = bosc_ame.tile.load_acc %c : memref<8x32xi32, #bosc_ame.spm<0>> -> vector<8x32xi32>

    // A: M = 8, K = 16, i8 phase (0x10010 = 65552), 16-byte row stride, and
    // the first tile of the MMA.
    // IR: call <vscale x 128 x i8> @llvm.riscv.bosc.ordered.load.a.nxv128i8(ptr {{.*}}, i64 16, i64 8, i64 16, i64 65552)
    %ta = bosc_ame.tile.load_a %a : memref<8x16xi8, strided<[16, 1], offset: 0>, #bosc_ame.spm<0>> -> vector<8x16xi8>

    // B: K = 16 and N = 32 come from the *logical* [K, N] shape, but the byte
    // stride is the physical row pitch 24, not strides[0] = 1.
    // IR: call <vscale x 128 x i8> @llvm.riscv.bosc.ordered.load.b.nxv128i8(ptr {{.*}}, i64 24, i64 32, i64 16, i64 65552)
    %tb = bosc_ame.tile.load_b %b : memref<16x32xi8, strided<[1, 24], offset: 0>, #bosc_ame.spm<1>> -> vector<16x32xi8>

    // M = 8, N = 32, K = 16 and the compute phase for the MMA itself.
    // IR: call <vscale x 32 x i32> @llvm.riscv.bosc.ordered.mma.nxv32i32.nxv128i8({{.*}}, i64 8, i64 32, i64 16, i64 65552)
    %c1 = bosc_ame.tile.mma %acc0, %ta, %tb : vector<8x32xi32>, vector<8x16xi8>, vector<16x32xi8> -> vector<8x32xi32>

    // The write-back programs the accumulator phase again and stores rows of
    // 128 bytes from the accumulator.
    // IR: call void @llvm.riscv.bosc.ordered.store.acc.nxv32i32(<vscale x 32 x i32> {{.*}}, ptr {{.*}}, i64 128, i64 8, i64 32, i64 65602)
    bosc_ame.tile.store_acc %c1, %c : vector<8x32xi32>, memref<8x32xi32, #bosc_ame.spm<0>>

    // No tile operation may fall through to the generic BOSCAME lowering, and
    // the SPM addressing must not go through an old memref-based AME op.  (The
    // named `bosc_ame.intr.*` operations exist, the `bosc_ame.tile.*` and
    // `bosc_ame.ml*.m` ones are exactly what must not survive.)
    // LOWER-NOT: "bosc_ame.intr.tile.
    // LOWER-NOT: "bosc_ame.mlae8.m"
    // LOWER-NOT: "bosc_ame.mlbe8.m"
    // LOWER-NOT: "bosc_ame.mlce32.m"
    // LOWER-NOT: "bosc_ame.msce32.m"
    // LOWER-NOT: "bosc_ame.mqma.b.mm"

    // Register allocation stays with the backend: the lowering names no
    // register and does not annotate a subtarget feature.
    return %c0_i32 : i32
  }
}
