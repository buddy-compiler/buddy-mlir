//
// x86
//
// RUN: buddy-opt %s -lower-dip="DIP-strip-mining=64" -arith-expand --convert-vector-to-scf --lower-affine --convert-scf-to-cf --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts  \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=i32 \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @global_input : memref<3x3xi8> = dense<[[97, 97, 97],
                                                                [97, 97, 97],
                                                                [97, 97, 97]]>

memref.global "private" @global_identity : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                   [0, 1, 0],
                                                                   [0, 0, 0]]>

memref.global "private" @global_output : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>

func.func private @printMemrefI8(memref<*xi8>) attributes { llvm.emit_c_interface }

func.func @main() -> i32 {
  %input = memref.get_global @global_input : memref<3x3xi8>
  %identity = memref.get_global @global_identity : memref<3x3xi8>
  %output = memref.get_global @global_output: memref<3x3xi8>

  %kernelAnchorX = arith.constant 1 : index
  %kernelAnchorY = arith.constant 1 : index
  %c = arith.constant 0 : i8 
  dip.corr_2d <CONSTANT_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, index, index, i8
  
  %printed_output = memref.cast %output : memref<3x3xi8> to memref<*xi8>
  call @printMemrefI8(%printed_output) : (memref<*xi8>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[3, 3\] strides = \[3, 1\] data =}}
  // a is ASCII for 97
  // CHECK{LITERAL}: [[a, a, a],
  // CHECK{LITERAL}: [a, a, a],
  // CHECK{LITERAL}: [a, a, a]]
  %ret = arith.constant 0 : i32 
  return %ret : i32
}
