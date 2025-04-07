// RUN: buddy-opt -matmul-parallel-vectorization-optimize -verify-diagnostics -expand-strided-metadata \
// RUN:    -lower-affine -convert-vector-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf \
// RUN:    -convert-linalg-to-loops -convert-scf-to-cf -convert-cf-to-llvm -llvm-request-c-wrappers -convert-func-to-llvm -convert-arith-to-llvm \
// RUN:    -reconcile-unrealized-casts %s \
// RUN: | mlir-runner -O0 -e buddy_matmul_i8 -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @A : memref<4x3xi8> = dense<[[9, 4, 6],[2, 4, 0],[6, 3, 3],[0, 4, 7]]>
memref.global "private" @B : memref<3x4xi8> = dense<[[1, 3, 8, 0],[1, 8, 8, 7], [6, 9, 7, 9]]>
memref.global "private" @C : memref<4x4xi8> = dense<[[49, 113, 46, 82],[6, 38, 48, 28],[24, 81, 36, 78],[8, 56, 0, 52]]>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @buddy_matmul_i8(){
  %a = memref.get_global @A : memref<4x3xi8>
  %b = memref.get_global @B : memref<3x4xi8>
  %c = memref.get_global @C : memref<4x4xi8>

  linalg.matmul
      ins(%a, %b: memref<4x3xi8>, memref<3x4xi8>)
      outs(%c: memref<4x4xi8>)

  %cst_0 = arith.constant 0 : index
  %cst_1 = arith.constant 1 : index
  %cst_4 = arith.constant 4 : index

  %c_f32 = memref.alloca() : memref<4x4xf32>
  scf.for %i = %cst_0 to %cst_4 step %cst_1 {
    scf.for %j = %cst_0 to %cst_4 step %cst_1 {
      %val_i8 = memref.load %c[%i, %j] : memref<4x4xi8>
      %val_f32 = arith.sitofp %val_i8 : i8 to f32
      memref.store %val_f32, %c_f32[%i, %j] : memref<4x4xf32>
    }
  }

  %printed_c = memref.cast %c_f32 : memref<4x4xf32> to memref<*xf32>
  call @printMemrefF32(%printed_c) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[4, 4\] strides = \[4, 1\] data =}}
  // CHECK{LITERAL}: [[98, -30, -64, -92],
  // CHECK{LITERAL}:  [12, 76, 96, 56],
  // CHECK{LITERAL}:  [51, -106, -127, 126],
  // CHECK{LITERAL}:  [54, -105, 81, -113]]
  return
}
