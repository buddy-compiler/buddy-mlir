// RUN: buddy-opt %s -split-input-file -verify-diagnostics

llvm.func @wrong_accumulator(%c: vector<[8]xi32>, %a: vector<[8]xi8>) {
  // expected-error @+1 {{result and accumulator must have the same type}}
  %r = "ime.intr.vmadot"(%c, %a, %a) : (vector<[8]xi32>, vector<[8]xi8>, vector<[8]xi8>) -> vector<[4]xi32>
  llvm.return
}

// -----

llvm.func @wrong_a100_inputs(%c: vector<[4]xi32>, %a: vector<[16]xi8>) {
  // expected-error @+1 {{invalid A100 dot tile types}}
  %r = "ime.intr.vmadot"(%c, %a, %a) : (vector<[4]xi32>, vector<[16]xi8>, vector<[16]xi8>) -> vector<[4]xi32>
  llvm.return
}

// -----

llvm.func @hp_bad_group(%c: vector<[4]xf16>, %a: vector<[8]xi8>) {
  // expected-error @+1 {{attribute 'group' failed to satisfy constraint}}
  %r = "ime.intr.vmadot.hp"(%c, %a, %a, %c) {group = 8 : i32} : (vector<[4]xf16>, vector<[8]xi8>, vector<[8]xi8>, vector<[4]xf16>) -> vector<[4]xf16>
  llvm.return
}

// -----

llvm.func @sp_bad_group(%c: vector<[4]xi32>, %a: vector<[16]xi8>, %b: vector<[8]xi8>) {
  // expected-error @+1 {{attribute 'group' failed to satisfy constraint}}
  %r = "ime.intr.vmadot.sp"(%c, %a, %b, %b) {group = 4 : i32} : (vector<[4]xi32>, vector<[16]xi8>, vector<[8]xi8>, vector<[8]xi8>) -> vector<[4]xi32>
  llvm.return
}

// -----

llvm.func @hp_wrong_pair(%c: vector<[8]xf16>, %a: vector<[8]xi8>) {
  // expected-error @+1 {{operand #0 must be A100 scalable tile vector}}
  %r = "ime.intr.vmadot.hp"(%c, %a, %a, %c) {group = 0 : i32} : (vector<[8]xf16>, vector<[8]xi8>, vector<[8]xi8>, vector<[8]xf16>) -> vector<[4]xf16>
  llvm.return
}

// -----

llvm.func @pack_wrong_width(%a: vector<[8]xi8>) {
  // expected-error @+1 {{pack/upack requires one-register inputs}}
  %r = "ime.intr.vpack"(%a, %a) {block = 0 : i32} : (vector<[8]xi8>, vector<[8]xi8>) -> vector<[8]xi16>
  llvm.return
}

// -----

llvm.func @pack_wrong_group(%a: vector<[8]xi8>) {
  // expected-error @+1 {{attribute 'block' failed to satisfy constraint}}
  %r = "ime.intr.vnpack4"(%a, %a) {block = -1 : i32} : (vector<[8]xi8>, vector<[8]xi8>) -> vector<[8]xi8>
  llvm.return
}

// -----

llvm.func @narrow_wrong_lanes(%a: vector<[4]xi16>) {
  // expected-error @+1 {{narrow pack requires one-register inputs}}
  %r = "ime.intr.vnpack"(%a, %a) {block = 0 : i32} : (vector<[4]xi16>, vector<[4]xi16>) -> vector<[4]xi8>
  llvm.return
}

// -----

llvm.func @nibble_wrong_type(%a: vector<[4]xi16>) {
  // expected-error @+1 {{4-bit pack requires vector<[8]xi8>}}
  %r = "ime.intr.vnpack4"(%a, %a) {block = 0 : i32} : (vector<[4]xi16>, vector<[4]xi16>) -> vector<[4]xi16>
  llvm.return
}
