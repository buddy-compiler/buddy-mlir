// RUN: buddy-opt %s -lower-xt-ame | FileCheck %s
// RUN: buddy-opt %s -lower-xt-ame -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir | FileCheck %s --check-prefix=LLVM
// RUN: buddy-opt %s -lower-xt-ame -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | buddy-translate --buddy-to-llvmir | buddy-llc -mtriple=riscv64 -mattr=+xtheadame -filetype=obj -o %t.o

func.func @mzero_store(%output: memref<4x4xi32>, %stride: i64) {
  %matrix = xt_ame.th.mzero : vector<4x4xi32>
  xt_ame.th.mste32 %matrix, %stride, %output : vector<4x4xi32>, memref<4x4xi32>
  return
}

// CHECK-LABEL: func.func @mzero_store
// CHECK: %[[MATRIX:.*]] = "xt_ame.intr.th.mzero"() : () -> vector<[16]xi32>
// CHECK: "xt_ame.intr.th.mste32"(%[[MATRIX]], %{{.*}}, %{{.*}}) : (vector<[16]xi32>, i64, !llvm.ptr) -> ()

// LLVM-LABEL: define void @mzero_store
// LLVM: call <vscale x 16 x i32> @llvm.riscv.th.mzero
// LLVM: call void @llvm.riscv.th.mste32
