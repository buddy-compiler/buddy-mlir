//
// x86
//
// RUN: buddy-opt %s -lower-dip="DIP-strip-mining=64" -arith-expand --convert-vector-to-scf --lower-affine --convert-scf-to-cf --convert-vector-to-llvm \
// RUN: --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts  \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=i32 \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @global_input : memref<3x3xi64> = dense<[[0 , 1 , 2 ],
                                                                 [10, 11, 12],
                                                                 [20, 21, 22]]>

memref.global "private" @global_identity : memref<3x3xi64> = dense<[[0, 0, 0],
                                                                    [0, 1, 0],
                                                                    [0, 0, 0]]>

memref.global "private" @global_output : memref<3x3xi64> = dense<[[0, 0, 0],
                                                                  [0, 0, 0],
                                                                  [0, 0, 0]]>

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @main() -> i32 {
  %input = memref.get_global @global_input : memref<3x3xi64>
  %identity = memref.get_global @global_identity : memref<3x3xi64>
  %output = memref.get_global @global_output: memref<3x3xi64>

  %kernelAnchorX = arith.constant 1 : index
  %kernelAnchorY = arith.constant 1 : index
  %c = arith.constant 0 : i64 
  dip.corr_2d <CONSTANT_PADDING> %input, %identity, %output, %kernelAnchorX, %kernelAnchorY, %c : memref<3x3xi64>, memref<3x3xi64>, memref<3x3xi64>, index, index, i64
  
  %printed_output = memref.cast %output : memref<3x3xi64> to memref<*xi64>
  call @printMemrefI64(%printed_output) : (memref<*xi64>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[3, 3\] strides = \[3, 1\] data =}}
  // CHECK{LITERAL}: [[0, 1, 2],
  // CHECK{LITERAL}: [10, 11, 12],
  // CHECK{LITERAL}: [20, 21, 22]]
  %ret = arith.constant 0 : i32
  return %ret : i32
}
