// RUN: buddy-opt -matmul-parallel-vectorization-optimize -verify-diagnostics -expand-strided-metadata -lower-affine \
// RUN:   -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:   -llvm-request-c-wrappers -convert-func-to-llvm -reconcile-unrealized-casts %s \
// RUN: | mlir-runner -O0 -e buddy_matmul_f32 -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @A : memref<4x3xf32> = dense<[[9., 4., 6.],[2., 4., 0.],[6., 3., 3.],[0., 4., 7.]]>
memref.global "private" @B : memref<3x4xf32> = dense<[[1., 3., 8., 0.],[1., 8., 8., 7.], [6., 9., 7., 9.]]>
memref.global "private" @C : memref<4x4xf32> = dense<[[49., 113., 146.,  82.],[6.,  38.,  48.,  28.],[24.,  81.,  36.,  78.],[8.,  56.,   0.,  52.]]>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @buddy_matmul_f32(){
  %a = memref.get_global @A : memref<4x3xf32>
  %b = memref.get_global @B : memref<3x4xf32>
  %c = memref.get_global @C : memref<4x4xf32>

  linalg.matmul
      ins(%a, %b: memref<4x3xf32>, memref<3x4xf32>)
      outs(%c: memref<4x4xf32>)
  %printed_c = memref.cast %c : memref<4x4xf32> to memref<*xf32>
  call @printMemrefF32(%printed_c) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[4, 4\] strides = \[4, 1\] data =}}
  // CHECK{LITERAL}: [[98, 226, 292, 164],
  // CHECK{LITERAL}:  [12, 76, 96, 56],
  // CHECK{LITERAL}:  [51, 150, 129, 126],
  // CHECK{LITERAL}:  [54, 151, 81, 143]]
  return
}
