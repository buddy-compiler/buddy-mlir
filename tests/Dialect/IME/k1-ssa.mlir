// RUN: buddy-translate -buddy-to-llvmir %s | buddy-llc -mtriple=riscv64 -mattr=+m,+v,+zvfh,+xsmtime -verify-machineinstrs -o - | FileCheck %s

// The corrected result/operand overload lists preserve the K1 intrinsic ABI.
llvm.func @k1_dot(%c: vector<[8]xi32>, %a: vector<[32]xi8>) -> vector<[8]xi32> {
  %r = "ime.intr.vmadot"(%c, %a, %a) : (vector<[8]xi32>, vector<[32]xi8>, vector<[32]xi8>) -> vector<[8]xi32>
  llvm.return %r : vector<[8]xi32>
}
// CHECK-LABEL: k1_dot:
// CHECK: vmadot

llvm.func @k1_fp(%c: vector<[16]xf16>, %a: vector<[16]xf16>) -> vector<[16]xf16> {
  %r = "ime.intr.vfmadot"(%c, %a, %a) : (vector<[16]xf16>, vector<[16]xf16>, vector<[16]xf16>) -> vector<[16]xf16>
  llvm.return %r : vector<[16]xf16>
}
// CHECK-LABEL: k1_fp:
// CHECK: vfmadot
