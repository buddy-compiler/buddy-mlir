// RUN: buddy-translate -buddy-to-llvmir %s | FileCheck %s
// RUN: buddy-translate -buddy-to-llvmir %s | buddy-llc -mtriple=riscv64 -mattr=+xtheadame -o - | FileCheck %s --check-prefix=ASM

// The intrinsic interface uses matrix values, independently of the high-level
// dialect's fixed-register operations. Exercise result and operand overloads.
llvm.func @integer_tiles(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) {
  %stride = llvm.mlir.constant(16 : i64) : i64
  "xt_ame.intr.th.mcfgmi"(%stride) : (i64) -> ()
  "xt_ame.intr.th.mcfgni"(%stride) : (i64) -> ()
  "xt_ame.intr.th.mcfgki"(%stride) : (i64) -> ()
  %zero = "xt_ame.intr.th.mzero"() : () -> vector<[16]xi32>
  %lhs = "xt_ame.intr.th.mlde8"(%stride, %a) : (i64, !llvm.ptr) -> vector<[64]xi8>
  %rhs = "xt_ame.intr.th.mldte8"(%stride, %b) : (i64, !llvm.ptr) -> vector<[64]xi8>
  %acc = "xt_ame.intr.th.mmacc.w.b"(%zero, %rhs, %lhs) : (vector<[16]xi32>, vector<[64]xi8>, vector<[64]xi8>) -> vector<[16]xi32>
  "xt_ame.intr.th.mste32"(%acc, %stride, %c) : (vector<[16]xi32>, i64, !llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: define void @integer_tiles
// CHECK: call void @llvm.riscv.th.mcfgmi.i64(i64 16)
// CHECK: %[[ZERO:.*]] = call <vscale x 16 x i32> @llvm.riscv.th.mzero.nxv16i32()
// CHECK: %[[LHS:.*]] = call <vscale x 64 x i8> @llvm.riscv.th.mlde8.nxv64i8.i64(i64 16, ptr %{{.*}})
// CHECK: %[[RHS:.*]] = call <vscale x 64 x i8> @llvm.riscv.th.mldte8.nxv64i8.i64(i64 16, ptr %{{.*}})
// CHECK: %[[ACC:.*]] = call <vscale x 16 x i32> @llvm.riscv.th.mmacc.w.b.nxv16i32.nxv64i8(<vscale x 16 x i32> %[[ZERO]], <vscale x 64 x i8> %[[RHS]], <vscale x 64 x i8> %[[LHS]])
// CHECK: call void @llvm.riscv.th.mste32.nxv16i32.i64(<vscale x 16 x i32> %[[ACC]], i64 16, ptr %{{.*}})
// ASM-LABEL: integer_tiles:
// ASM: th.mzero m{{[0-7]}}
// ASM: th.mlde8 m{{[0-7]}}
// ASM: th.mldte8 m{{[0-7]}}
// ASM: th.mmacc.w.b m{{[0-7]}}, m{{[0-7]}}, m{{[0-7]}}
// ASM: th.mste32 m{{[0-7]}}

llvm.func @float_tiles(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) {
  %stride = llvm.mlir.constant(32 : i64) : i64
  %zero = "xt_ame.intr.th.mzero"() : () -> vector<[16]xf32>
  %lhs = "xt_ame.intr.th.mlde32"(%stride, %a) : (i64, !llvm.ptr) -> vector<[16]xf32>
  %rhs = "xt_ame.intr.th.mldte32"(%stride, %b) : (i64, !llvm.ptr) -> vector<[16]xf32>
  %acc = "xt_ame.intr.th.mfmacc.s"(%zero, %rhs, %lhs) : (vector<[16]xf32>, vector<[16]xf32>, vector<[16]xf32>) -> vector<[16]xf32>
  "xt_ame.intr.th.mste32"(%acc, %stride, %c) : (vector<[16]xf32>, i64, !llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: define void @float_tiles
// CHECK: %[[ZERO:.*]] = call <vscale x 16 x float> @llvm.riscv.th.mzero.nxv16f32()
// CHECK: %[[LHS:.*]] = call <vscale x 16 x float> @llvm.riscv.th.mlde32.nxv16f32.i64
// CHECK: %[[RHS:.*]] = call <vscale x 16 x float> @llvm.riscv.th.mldte32.nxv16f32.i64
// CHECK: %[[ACC:.*]] = call <vscale x 16 x float> @llvm.riscv.th.mfmacc.s.m.nxv16f32(<vscale x 16 x float> %[[ZERO]], <vscale x 16 x float> %[[RHS]], <vscale x 16 x float> %[[LHS]])
// CHECK: call void @llvm.riscv.th.mste32.nxv16f32.i64(<vscale x 16 x float> %[[ACC]], i64 32, ptr %{{.*}})
// ASM-LABEL: float_tiles:
// ASM: th.mfmacc.s m{{[0-7]}}, m{{[0-7]}}, m{{[0-7]}}

llvm.func @widen_float_tiles(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) {
  %stride = llvm.mlir.constant(32 : i64) : i64
  %zero = "xt_ame.intr.th.mzero"() : () -> vector<[8]xf64>
  %lhs = "xt_ame.intr.th.mlde32"(%stride, %a) : (i64, !llvm.ptr) -> vector<[16]xf32>
  %rhs = "xt_ame.intr.th.mldte32"(%stride, %b) : (i64, !llvm.ptr) -> vector<[16]xf32>
  %acc = "xt_ame.intr.th.mfmacc.d.s"(%zero, %rhs, %lhs) : (vector<[8]xf64>, vector<[16]xf32>, vector<[16]xf32>) -> vector<[8]xf64>
  "xt_ame.intr.th.mste64"(%acc, %stride, %c) : (vector<[8]xf64>, i64, !llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: define void @widen_float_tiles
// CHECK: call <vscale x 8 x double> @llvm.riscv.th.mfmacc.d.s.m.nxv8f64.nxv16f32
// CHECK: call void @llvm.riscv.th.mste64.nxv8f64.i64
// ASM-LABEL: widen_float_tiles:
// ASM: th.mfmacc.d.s m{{[0-7]}}, m{{[0-7]}}, m{{[0-7]}}
