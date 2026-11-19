// RUN: buddy-opt %s --split-input-file --verify-diagnostics

// Ordinary operations reject mismatched register banks before LLVM lowering.
!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
!a = !llvm.target<"riscv.ztt.acc", i32, 8, 8>

llvm.func @matrix_destination_in_acc_bank(%dst: !a, %src: !m) {
  // expected-error @+1 {{operand #0 must be}}
  %result = "ame.mabs.ew"(%dst, %src) : (!a, !m) -> !m
  llvm.return
}

// -----

!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>

llvm.func @acc_destination_in_matrix_bank(%dst: !m, %lhs: !m, %rhs: !m) {
  // expected-error @+1 {{operand #0 must be}}
  %result = "ame.mmulacc.2d"(%dst, %lhs, %rhs) : (!m, !m, !m) -> !m
  llvm.return
}

// -----

llvm.func @floating_scalar(%request: f32) {
  // expected-error @+1 {{operand #0 must be}}
  %result = "ame.acquire"(%request) : (f32) -> f32
  llvm.return
}

// -----

llvm.func @oversized_scalar(%request: i128) {
  // expected-error @+1 {{operand #0 must be}}
  %result = "ame.acquire"(%request) : (i128) -> i128
  llvm.return
}

// -----

!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>

llvm.func @nondefault_pointer(%dst: !m, %base: !llvm.ptr<1>) {
  // expected-error @+1 {{operand #1 must be}}
  %result = "ame.mls.rm"(%dst, %base) : (!m, !llvm.ptr<1>) -> !m
  llvm.return
}

// -----

!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
!m8 = !llvm.target<"riscv.ztt.matrix", i8, 8, 8>

llvm.func @mismatched_passthrough(%dst: !m, %src: !m8) {
  // expected-error @+1 {{failed to verify that all of {md_in, res} have same type}}
  %result = "ame.mconv.ew"(%dst, %src) : (!m, !m8) -> !m8
  llvm.return
}

// -----

llvm.func @mismatched_acquire_result(%request: i32) {
  // expected-error @+1 {{failed to verify that all of {rs1, res} have same type}}
  %result = "ame.acquire"(%request) : (i32) -> i64
  llvm.return
}

// -----

!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>

llvm.func @mismatched_extract_index_result(%matrix: !m, %index: i64) {
  // expected-error @+1 {{failed to verify that all of {rs1, res} have same type}}
  %result = "ame.mmove32.x.m"(%matrix, %index) : (!m, i64) -> i32
  llvm.return
}

// -----

!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
!m8 = !llvm.target<"riscv.ztt.matrix", i8, 8, 8>
!pair = !llvm.struct<(!m, !m8)>

llvm.func @colzip_mismatched_fields(%lhs: !m, %rhs: !m) {
  // expected-error @+1 {{requires a literal, unpacked two-matrix result matching both operands}}
  %result = "ame.mcolzip.ew"(%lhs, %rhs) : (!m, !m) -> !pair
  llvm.return
}

// -----

!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
!m8 = !llvm.target<"riscv.ztt.matrix", i8, 8, 8>
!pair = !llvm.struct<(!m, !m)>

llvm.func @colunzip_mismatched_sources(%lhs: !m, %rhs: !m8) {
  // expected-error @+1 {{requires a literal, unpacked two-matrix result matching both operands}}
  %result = "ame.mcolunzip.ew"(%lhs, %rhs) : (!m, !m8) -> !pair
  llvm.return
}

// -----

!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
!pair = !llvm.struct<packed (!m, !m)>

llvm.func @rowzip_packed_result(%lhs: !m, %rhs: !m) {
  // expected-error @+1 {{requires a literal, unpacked two-matrix result matching both operands}}
  %result = "ame.mrowzip.ew"(%lhs, %rhs) : (!m, !m) -> !pair
  llvm.return
}

// -----

!m = !llvm.target<"riscv.ztt.matrix", i32, 8, 8>
!pair = !llvm.struct<(!m)>

llvm.func @rowunzip_single_field(%lhs: !m, %rhs: !m) {
  // expected-error @+1 {{requires a literal, unpacked two-matrix result matching both operands}}
  %result = "ame.mrowunzip.ew"(%lhs, %rhs) : (!m, !m) -> !pair
  llvm.return
}
