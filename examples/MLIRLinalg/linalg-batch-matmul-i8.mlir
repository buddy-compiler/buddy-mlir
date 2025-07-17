// RUN: buddy-opt -batchmatmul-optimize -verify-diagnostics -expand-strided-metadata -lower-affine \
// RUN:   -convert-vector-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf -convert-linalg-to-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm \
// RUN:   -llvm-request-c-wrappers -convert-func-to-llvm -reconcile-unrealized-casts %s \
// RUN: | mlir-runner -O0 -e buddy_batchmatmul_i8 -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @A : memref<2x2x3xi8> = dense<[[[9, 4, 6],[2, 4, 0]],[[6, 3, 3],[0, 4, 7]]]>
memref.global "private" @B : memref<2x3x4xi8> = dense<[[[1, 3, 8, 0],[1, 8, 8, 7], [6, 9, 7, 9]],[[3, 8, 6, 8],[2, 7, 0, 6],[0, 4, 0, 4]]]>
memref.global "private" @C : memref<2x2x4xi8> = dense<[[[49, 12, 14, 82],[6, 38, 48, 28]],[[24, 81, 36, 78],[8, 56, 0, 52]]]>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @buddy_batchmatmul_i8(){
  %a = memref.get_global @A : memref<2x2x3xi8>
  %b = memref.get_global @B : memref<2x3x4xi8>
  %c = memref.get_global @C : memref<2x2x4xi8>

  linalg.batch_matmul
      ins(%a, %b: memref<2x2x3xi8>, memref<2x3x4xi8>)
      outs(%c: memref<2x2x4xi8>)

  %cst_0 = arith.constant 0 : index
  %cst_1 = arith.constant 1 : index
  %cst_2 = arith.constant 2 : index
  %cst_4 = arith.constant 4 : index

  %c_f32 = memref.alloca() : memref<2x2x4xf32>
  scf.for %i = %cst_0 to %cst_2 step %cst_1 {
    scf.for %j = %cst_0 to %cst_2 step %cst_1 {
      scf.for %k = %cst_0 to %cst_4 step %cst_1 {
        %val_i8 = memref.load %c[%i, %j, %k] : memref<2x2x4xi8>
        %val_f32 = arith.sitofp %val_i8 : i8 to f32
        memref.store %val_f32, %c_f32[%i, %j, %k] : memref<2x2x4xf32>
      }
    }
  }

  %printed_c = memref.cast %c_f32 : memref<2x2x4xf32> to memref<*xf32>
  call @printMemrefF32(%printed_c) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 3 offset = 0 sizes = \[2, 2, 4\] strides = \[8, 4, 1\] data =}}
  // CHECK{LITERAL}: [[[98,    125,    -96,    -92],
  // CHECK{LITERAL}:   [12,    76,    96,    56]],
  // CHECK{LITERAL}:  [[48,    -94,    72,    -100],
  // CHECK{LITERAL}:   [16,    112,    0,    104]]]
  return
}
