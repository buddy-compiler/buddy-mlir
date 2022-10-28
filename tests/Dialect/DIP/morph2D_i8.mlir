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

memref.global "private" @global_outputerosion : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>
memref.global "private" @global_outputdilation : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>
memref.global "private" @global_outputopening : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>
memref.global "private" @global_outputinter : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>
memref.global "private" @global_outputclosing : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>
memref.global "private" @global_outputtophat : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>
memref.global "private" @global_outputinter1 : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>
memref.global "private" @global_inputinter : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>
memref.global "private" @global_outputbottomhat : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>

memref.global "private" @global_outputbottomhatinter1 : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>
memref.global "private" @global_inputbottomhatinter : memref<3x3xi8> = dense<[[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]>

memref.global "private" @global_kernel : memref<3x3xi8> = dense<[[12, 22, 33],
                                                                    [45, 44, 0],
                                                                    [90, 11, 10]]>

memref.global "private" @global_kernel1 : memref<3x3xi8> = dense<[[0, 0, 11],
                                                                    [4, 44, 10],
                                                                    [9, 100, 10]]>

memref.global "private" @global_kernel2 : memref<3x3xi8> = dense<[[1, 0, 0],
                                                                    [0, 225, 0],
                                                                    [0, 11, 10]]>

memref.global "private" @global_kernel3 : memref<3x3xi8> = dense<[[100, 0, 0],
                                                                    [0, 0, 110],
                                                                    [190, 0, 0]]>
memref.global "private" @global_copymemref1 : memref<3x3xi8> = dense<[[-1, -1, -1],
                                                                    [-1, -1, -1],
                                                                    [-1, -1, -1]]>

memref.global "private" @global_copymemref2 : memref<3x3xi8> = dense<[[255, 255, 255],
                                                                    [255, 255, 255],
                                                                    [255, 255, 255]]>



func.func private @printMemrefI8(memref<*xi8>) attributes { llvm.emit_c_interface }

func.func @main() -> i32 {
  %input = memref.get_global @global_input : memref<3x3xi8>
  %identity = memref.get_global @global_identity : memref<3x3xi8>
  %kernel = memref.get_global @global_kernel : memref<3x3xi8>
  %kernel1 = memref.get_global @global_kernel1 : memref<3x3xi8>
  %kernel2 = memref.get_global @global_kernel2 : memref<3x3xi8>
  %kernel3 = memref.get_global @global_kernel3 : memref<3x3xi8>
  %outputErosion = memref.get_global @global_outputerosion: memref<3x3xi8>
  %outputDilation = memref.get_global @global_outputdilation : memref<3x3xi8>
  %outputOpening = memref.get_global @global_outputopening : memref<3x3xi8>
  %outputClosing = memref.get_global @global_outputclosing : memref<3x3xi8>
  %outputTopHat = memref.get_global @global_outputtophat : memref<3x3xi8>
  %outputBottomHat = memref.get_global @global_outputbottomhat : memref<3x3xi8>
  %inputInter = memref.get_global @global_inputinter : memref<3x3xi8>
  %outputInter = memref.get_global @global_outputinter : memref<3x3xi8>
  %outputInter1 = memref.get_global @global_outputinter1 : memref<3x3xi8>
  %copyMemref1 = memref.get_global @global_copymemref1 : memref<3x3xi8>
  %copyMemref2 = memref.get_global @global_copymemref2 : memref<3x3xi8>


  %kernelAnchorX = arith.constant 1 : index
  %kernelAnchorY = arith.constant 1 : index
  %iterations = arith.constant 1 : index
  %c = arith.constant 0 : i8

  dip.erosion_2d <CONSTANT_PADDING> %input, %identity, %outputErosion, %copyMemref2, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, index, index, index, i8
  dip.dilation_2d <REPLICATE_PADDING> %input, %kernel, %outputDilation, %copyMemref1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, index, index, index, i8
  dip.opening_2d <CONSTANT_PADDING> %input, %kernel2, %outputOpening, %outputInter, %copyMemref2, %copyMemref1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, index, index, index, i8
  dip.closing_2d <CONSTANT_PADDING> %input, %kernel3, %outputClosing, %outputInter, %copyMemref1, %copyMemref2, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, index, index, index, i8
  dip.tophat_2d <REPLICATE_PADDING> %input, %kernel1,%outputTopHat, %outputInter, %outputInter1, %inputInter, %copyMemref2, %copyMemref1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, index, index, index, i8
  dip.bottomhat_2d <CONSTANT_PADDING> %input, %kernel2, %outputBottomHat, %outputInter, %outputInter1, %inputInter, %copyMemref1, %copyMemref2, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, memref<3x3xi8>, index, index, index, i8

  %printed_outpute = memref.cast %outputErosion : memref<3x3xi8> to memref<*xi8>
  %printed_outputd = memref.cast %outputDilation : memref<3x3xi8> to memref<*xi8>
  %printed_outputo = memref.cast %outputOpening : memref<3x3xi8> to memref<*xi8>
  %printed_outputc = memref.cast %outputClosing : memref<3x3xi8> to memref<*xi8>
  %printed_outputt = memref.cast %outputTopHat : memref<3x3xi8> to memref<*xi8>
  %printed_outputb = memref.cast %outputBottomHat : memref<3x3xi8> to memref<*xi8>
  call @printMemrefI8(%printed_outpute) : (memref<*xi8>) -> ()
  call @printMemrefI8(%printed_outputd) : (memref<*xi8>) -> ()
  call @printMemrefI8(%printed_outputo) : (memref<*xi8>) -> ()
  call @printMemrefI8(%printed_outputc) : (memref<*xi8>) -> ()
  call @printMemrefI8(%printed_outputt) : (memref<*xi8>) -> ()
  call @printMemrefI8(%printed_outputb) : (memref<*xi8>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[3, 3\] strides = \[3, 1\] data =}}
  // a is ASCII for 97
  // CHECK{LITERAL}: [[a, a, a],
  // CHECK{LITERAL}: [a, a, a],
  // CHECK{LITERAL}: [a, a, a]]
  %ret = arith.constant 0 : i32
  return %ret : i32
}
