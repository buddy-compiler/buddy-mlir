define void @main() {
entry:
  call void @llvm.riscv.mvin(i64 1000, i64 10000)
  ret void
}

declare void @llvm.riscv.mvin(i64, i64)
