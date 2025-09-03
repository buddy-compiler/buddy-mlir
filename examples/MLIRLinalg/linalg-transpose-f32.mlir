// RUN: buddy-opt -transpose-optimize="vector-size=16" -verify-diagnostics -lower-affine -expand-strided-metadata -convert-vector-to-scf -convert-vector-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -lower-affine -llvm-request-c-wrappers -convert-arith-to-llvm -reconcile-unrealized-casts %s \
// RUN: | mlir-runner -O0 -e buddy_transpose_f32 -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @A : memref<3x4xf32> = dense<[[1., 3., 8., 0.],[1., 8., 8., 7.], [6., 9., 7., 9.]]>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @buddy_transpose_f32(){
  %a = memref.get_global @A : memref<3x4xf32>
  %b = memref.alloc() : memref<4x3xf32>

  linalg.transpose
      ins(%a: memref<3x4xf32>)
      outs(%b: memref<4x3xf32>)
      permutation = [1, 0]
  %printed_b = memref.cast %b : memref<4x3xf32> to memref<*xf32>
  call @printMemrefF32(%printed_b) : (memref<*xf32>) -> ()
  memref.dealloc %b : memref<4x3xf32>
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[4, 3\] strides = \[3, 1\] data =}}
  // CHECK{LITERAL}: [[1, 1, 6],
  // CHECK{LITERAL}:  [3, 8, 9],
  // CHECK{LITERAL}:  [8, 8, 7],
  // CHECK{LITERAL}:  [0, 7, 9]]
  return
}
