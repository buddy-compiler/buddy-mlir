// RUN: not buddy-opt %s '--lower-linalg-to-boscame=triton-w8a8-fast-path=true' 2>&1 | \
// RUN:   FileCheck %s --check-prefix=FASTPATH
// RUN: not buddy-opt %s --lower-qwen-w8a8-to-boscame 2>&1 | \
// RUN:   FileCheck %s --check-prefix=W8A8
//
// The fast-path switch and the FPGA-only W8A8 pass select optimisations, not the
// hardware contract, so neither may be the thing that switches the AME encoding:
// without an explicit FPGA target they are rejected instead of silently emitting
// the bit-field configuration or the FPGA final-store-convert (plan A.2/A.3).
//
// FASTPATH: error: triton-w8a8-fast-path requires bosc_ame.target = "qwen3-fpga"
// W8A8: error: --lower-qwen-w8a8-to-boscame requires bosc_ame.target = "qwen3-fpga"
module {
  func.func @placeholder(%A: memref<4x4xi32>) {
    return
  }
}
