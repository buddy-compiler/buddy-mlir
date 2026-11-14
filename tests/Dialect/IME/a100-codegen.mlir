// RUN: %PYTHON %S/../../../examples/IMEDialect/generate_a100_tests.py %t
// RUN: buddy-opt %t/a100-kernels.mlir -lower-ime="target=k3" | buddy-translate -buddy-to-llvmir > %t/kernels.ll
// RUN: FileCheck %s --check-prefix=IR < %t/kernels.ll
// RUN: buddy-llc -mtriple=riscv64 -mcpu=spacemit-a100 -verify-machineinstrs %t/kernels.ll -o - | FileCheck %s --check-prefix=ASM

// All 117 supported non-window type/immediate combinations must select.
// HP/SP use fixed signatures; pack/upack overload on the result, while narrow
// packs overload on the input. Accumulators are SSA results, never void calls.
// IR-DAG: call <vscale x 4 x i32> @llvm.riscv.ime.vmadot.nxv4i32.nxv8i8.nxv8i8
// IR-DAG: call <vscale x 4 x half> @llvm.riscv.ime.vmadot.hp({{.*}}i32 7)
// IR-DAG: call <vscale x 4 x i32> @llvm.riscv.ime.vmadot.sp({{.*}}i32 3)
// IR-DAG: call <vscale x 4 x float> @llvm.riscv.ime.vfmadot.nxv4f32.nxv4f16.nxv4f16
// IR-DAG: call <vscale x 16 x i8> @llvm.riscv.ime.vpack.nxv16i8
// IR-DAG: call <vscale x 2 x i64> @llvm.riscv.ime.vupack.nxv2i64
// IR-DAG: call <vscale x 8 x i8> @llvm.riscv.ime.vnpack.nxv4i16
// IR-DAG: call <vscale x 2 x i32> @llvm.riscv.ime.vnspack.nxv1i64
// IR-DAG: call <vscale x 8 x i8> @llvm.riscv.ime.vnpack4
// IR-DAG: call <vscale x 8 x i8> @llvm.riscv.ime.vnspack4

// Whole-register loads do not need an e8/m2 configuration before each dot.
// ASM-LABEL: a100_vmadot_8_0:
// ASM: vl2re32.v
// ASM: vl1r.v
// ASM: vsetvli {{.*}}e32, m1
// ASM: smt.vmadot {{.*}}i8
// ASM-NEXT: smt.vmadot {{.*}}i8
// ASM: vs2r.v

// ASM-DAG: smt.vmadot.hp {{.*}}, v{{[01]}}, 7, i8
// ASM-DAG: smt.vmadot.sp {{.*}}, v{{[01]}}, 3, i8
// ASM-DAG: smt.vmadotu {{.*}}i8
// ASM-DAG: smt.vmadotu.hp
// ASM-DAG: smt.vmadotu.sp
// ASM-DAG: smt.vmadotsu {{.*}}i8
// ASM-DAG: smt.vmadotsu.hp
// ASM-DAG: smt.vmadotsu.sp
// ASM-DAG: smt.vmadotus {{.*}}i8
// ASM-DAG: smt.vmadotus.hp
// ASM-DAG: smt.vmadotus.sp
// ASM-DAG: smt.vfwmadot
// ASM-DAG: smt.vpack.vv
// ASM-DAG: smt.vupack.vv
// ASM-DAG: smt.vnpack.vv
// ASM-DAG: smt.vnspack.vv
// ASM-DAG: smt.vnpack4.vv
// ASM-DAG: smt.vnspack4.vv

module {}
