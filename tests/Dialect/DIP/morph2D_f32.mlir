//
// x86
//
// RUN: buddy-opt %s -lower-dip="DIP-strip-mining=64" -arith-expand --convert-vector-to-scf --lower-affine --convert-scf-to-cf --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts  \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=i32 \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @global_input : memref<3x3xf32> = dense<[[0. , 1. , 2. ],
                                                                 [10., 11., 12.],
                                                                 [20., 21., 22.]]>

memref.global "private" @global_identity : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                    [0., 1., 0.],
                                                                    [0., 0., 0.]]>

memref.global "private" @global_outputerosion : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputdilation : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputopening : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputopeninginter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputclosing : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputclosinginter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputtophat : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputtophatinter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputtophatinter1 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>

memref.global "private" @global_inputtophatinter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputbottomhat : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputbottomhatinter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>
memref.global "private" @global_outputbottomhatinter1 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>

memref.global "private" @global_inputbottomhatinter : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                  [0., 0., 0.],
                                                                  [0., 0., 0.]]>

memref.global "private" @global_kernel : memref<3x3xf32> = dense<[[12., 22., 33.],
                                                                    [45., 44., 0.],
                                                                    [90., 11., 10.]]>

memref.global "private" @global_kernel1 : memref<3x3xf32> = dense<[[0., 0., 11.],
                                                                    [4., 44., 10.],
                                                                    [9., 100., 10.]]>

memref.global "private" @global_kernel2 : memref<3x3xf32> = dense<[[1., 0., 0.],
                                                                    [0., 225., 0.],
                                                                    [0., 11., 10.]]>

memref.global "private" @global_kernel3 : memref<3x3xf32> = dense<[[100., 0., 0.],
                                                                    [0., 0., 110.],
                                                                    [190., 0., 0.]]>

memref.global "private" @global_copymemref1 : memref<3x3xf32> = dense<[[-1., -1., -1.],
                                                                    [-1., -1., -1.],
                                                                    [-1., -1., -1.]]>

memref.global "private" @global_copymemref2 : memref<3x3xf32> = dense<[[256., 256., 256.],
                                                                    [256., 256., 256.],
                                                                    [256., 256., 256.]]>



func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @main() -> i32 {
  %input = memref.get_global @global_input : memref<3x3xf32>
  %identity = memref.get_global @global_identity : memref<3x3xf32>
  %kernel = memref.get_global @global_kernel : memref<3x3xf32>
  %kernel1 = memref.get_global @global_kernel1 : memref<3x3xf32>
  %kernel2 = memref.get_global @global_kernel2 : memref<3x3xf32>
  %kernel3 = memref.get_global @global_kernel3 : memref<3x3xf32>
  %outpute = memref.get_global @global_outputerosion : memref<3x3xf32>
  %outputd = memref.get_global @global_outputerosion : memref<3x3xf32>
  %outputOpening = memref.get_global @global_outputopening : memref<3x3xf32>
  %outputOpening1 = memref.get_global @global_outputopeninginter : memref<3x3xf32>
  %outputClosing = memref.get_global @global_outputclosing : memref<3x3xf32>
  %outputClosing1 = memref.get_global @global_outputclosinginter : memref<3x3xf32>
  %outputTopHat = memref.get_global @global_outputtophat : memref<3x3xf32>
  %outputTopHat1 = memref.get_global @global_outputtophatinter : memref<3x3xf32>
  %outputTopHat2 = memref.get_global @global_outputtophatinter1 : memref<3x3xf32>
  %inputTopHat1 = memref.get_global @global_inputtophatinter : memref<3x3xf32>
  %outputBottomHat = memref.get_global @global_outputbottomhat : memref<3x3xf32>
  %outputBottomHat1 = memref.get_global @global_outputbottomhatinter : memref<3x3xf32>
  %outputBottomHat2 = memref.get_global @global_outputbottomhatinter1 : memref<3x3xf32>
  %inputBottomHat1 = memref.get_global @global_inputbottomhatinter : memref<3x3xf32>
  %copyMemref1 = memref.get_global @global_copymemref1 : memref<3x3xf32>
  %copyMemref2 = memref.get_global @global_copymemref2 : memref<3x3xf32>

  %kernelAnchorX = arith.constant 1 : index
  %kernelAnchorY = arith.constant 1 : index
  %iterations = arith.constant 1 : index
  %c = arith.constant 0. : f32

  dip.erosion_2d <CONSTANT_PADDING> %input, %identity, %outpute, %copyMemref2, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  dip.dilation_2d <REPLICATE_PADDING> %input, %kernel, %outputd, %copyMemref1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  dip.opening_2d <CONSTANT_PADDING> %input, %kernel1, %outputOpening, %outputOpening1, %copyMemref2, %copyMemref1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>,memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  dip.closing_2d <REPLICATE_PADDING> %input, %kernel2, %outputClosing, %outputClosing1, %copyMemref1, %copyMemref2, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  dip.tophat_2d <REPLICATE_PADDING> %input, %kernel3, %outputTopHat, %outputTopHat1,%outputTopHat2, %inputTopHat1, %copyMemref2, %copyMemref1, %kernelAnchorX, %kernelAnchorY, %iterations, %c : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  dip.bottomhat_2d <CONSTANT_PADDING> %input, %kernel, %outputBottomHat, %outputBottomHat1,%outputBottomHat2, %inputBottomHat1, %copyMemref1, %copyMemref2, %kernelAnchorX, %kernelAnchorY, %iterations, %c: memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32

  %printed_outpute = memref.cast %outpute : memref<3x3xf32> to memref<*xf32>
  %printed_outputd = memref.cast %outputd : memref<3x3xf32> to memref<*xf32>
  %printed_outputo = memref.cast %outputOpening : memref<3x3xf32> to memref<*xf32>
  %printed_outputc = memref.cast %outputClosing : memref<3x3xf32> to memref<*xf32>
  %printed_outputt = memref.cast %outputTopHat : memref<3x3xf32> to memref<*xf32>
  %printed_outputb = memref.cast %outputBottomHat : memref<3x3xf32> to memref<*xf32>

  call @printMemrefF32(%printed_outpute) : (memref<*xf32>) -> ()
  call @printMemrefF32(%printed_outputd) : (memref<*xf32>) -> ()
  call @printMemrefF32(%printed_outputo) : (memref<*xf32>) -> ()
  call @printMemrefF32(%printed_outputc) : (memref<*xf32>) -> ()
  call @printMemrefF32(%printed_outputt) : (memref<*xf32>) -> ()
  call @printMemrefF32(%printed_outputb) : (memref<*xf32>) -> ()
  // CHECK: {{Unranked Memref base@ = 0x[0-9A-Fa-f]{1,} rank = 2 offset = 0 sizes = \[3, 3\] strides = \[3, 1\] data =}}
  // CHECK{LITERAL}: [[0, 1, 2],
  // CHECK{LITERAL}: [10, 11, 12],
  // CHECK{LITERAL}: [20, 21, 22]]

  %ret = arith.constant 0 : i32
  return %ret : i32
}
