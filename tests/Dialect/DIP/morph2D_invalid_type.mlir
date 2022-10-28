//
// x86
//
// RUN: buddy-opt %s -lower-dip="DIP-strip-mining=64" -arith-expand --convert-vector-to-scf --lower-affine --convert-scf-to-cf --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts 2>&1 | FileCheck %s

memref.global "private" @global_input_f32 : memref<3x3xf32> = dense<[[0. , 1. , 2. ],
                                                                     [10., 11., 12.],
                                                                     [20., 21., 22.]]>

memref.global "private" @global_identity_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>

memref.global "private" @global_output_erosion_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>
memref.global "private" @global_output_dilation_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>

memref.global "private" @global_output_opening_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>
memref.global "private" @global_output_openinginter_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>

memref.global "private" @global_output_closing_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>
memref.global "private" @global_output_closinginter_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>

memref.global "private" @global_input_i32 : memref<3x3xi32> = dense<[[0 , 1 , 2 ],
                                                                     [10, 11, 12],
                                                                     [20, 21, 22]]>

memref.global "private" @global_identity_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                        [0, 1, 0],
                                                                        [0, 0, 0]]>

memref.global "private" @global_output_erosion_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>

memref.global "private" @global_output_dilation_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>
memref.global "private" @global_output_opening_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>

memref.global "private" @global_output_openinginter_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>
memref.global "private" @global_output_closing_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>

memref.global "private" @global_output_closinginter_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>

memref.global "private" @global_input_f128 : memref<3x3xf128> = dense<[[0. , 1. , 2. ],
                                                                       [10., 11., 12.],
                                                                       [20., 21., 22.]]>

memref.global "private" @global_output_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_erosion_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_dilation_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_opening_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_openinginter_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_closing_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>
memref.global "private" @global_output_closinginter_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>

memref.global "private" @global_identity_f128 : memref<3x3xf128> = dense<[[0., 0., 0.],
                                                                          [0., 0., 0.],
                                                                          [0., 0., 0.]]>

memref.global "private" @global_copymemref1_f128 : memref<3x3xf128> = dense<[[-1., -1., -1.],
                                                                    [-1., -1., -1.],
                                                                    [-1., -1., -1.]]>

memref.global "private" @global_copymemref2_f128 : memref<3x3xf128> = dense<[[256., 256., 256.],
                                                                    [256., 256., 256.],
                                                                    [256., 256., 256.]]>

memref.global "private" @global_copymemref1_f32 : memref<3x3xf32> = dense<[[-1., -1., -1.],
                                                                    [-1., -1., -1.],
                                                                    [-1., -1., -1.]]>

memref.global "private" @global_copymemref2_f32 : memref<3x3xf32> = dense<[[256., 256., 256.],
                                                                    [256., 256., 256.],
                                                                    [256., 256., 256.]]>

memref.global "private" @global_copymemref1_i32 : memref<3x3xi32> = dense<[[-1, -1, -1],
                                                                    [-1, -1, -1],
                                                                    [-1, -1, -1]]>

memref.global "private" @global_copymemref2_i32 : memref<3x3xi32> = dense<[[256, 256, 256],
                                                                    [256, 256, 256],
                                                                    [256, 256, 256]]>

func.func @main() -> i32 {
  %input_f32 = memref.get_global @global_input_f32 : memref<3x3xf32>
  %identity_f32 = memref.get_global @global_identity_f32 : memref<3x3xf32>
  %output_f32Erosion = memref.get_global @global_output_erosion_f32 : memref<3x3xf32>
  %output_f32Dilation = memref.get_global @global_output_dilation_f32 : memref<3x3xf32>
  %output_f32Opening = memref.get_global @global_output_opening_f32 : memref<3x3xf32>
  %output_f32Opening1 = memref.get_global @global_output_openinginter_f32 : memref<3x3xf32>
  %output_f32Closing = memref.get_global @global_output_closing_f32 : memref<3x3xf32>
  %output_f32Closing1 = memref.get_global @global_output_closinginter_f32 : memref<3x3xf32>
  %copyMemreff32_1 = memref.get_global @global_copymemref1_f32 : memref<3x3xf32>
  %copyMemreff32_2 = memref.get_global @global_copymemref2_f32 : memref<3x3xf32>
  %c_f32 = arith.constant 0. : f32

  %input_i32 = memref.get_global @global_input_i32 : memref<3x3xi32>
  %identity_i32 = memref.get_global @global_identity_i32 : memref<3x3xi32>
  %output_i32Erosion = memref.get_global @global_output_erosion_i32 : memref<3x3xi32>
  %output_i32Dilation = memref.get_global @global_output_dilation_i32 : memref<3x3xi32>
  %output_i32Opening = memref.get_global @global_output_opening_i32 : memref<3x3xi32>
  %output_i32Opening1 = memref.get_global @global_output_openinginter_i32 : memref<3x3xi32>
  %output_i32Closing = memref.get_global @global_output_closing_i32 : memref<3x3xi32>
  %output_i32Closing1 = memref.get_global @global_output_closinginter_i32 : memref<3x3xi32>
  %copyMemrefi32_1 = memref.get_global @global_copymemref1_i32 : memref<3x3xi32>
  %copyMemrefi32_2 = memref.get_global @global_copymemref2_i32 : memref<3x3xi32>
  %c_i32 = arith.constant 0 : i32

  %input_f128 = memref.get_global @global_input_f128 : memref<3x3xf128>
  %identity_f128 = memref.get_global @global_identity_f128 : memref<3x3xf128>
  %output_f128Erosion = memref.get_global @global_output_erosion_f128 : memref<3x3xf128>
  %output_f128Dilation = memref.get_global @global_output_dilation_f128 : memref<3x3xf128>
  %output_f128Opening = memref.get_global @global_output_opening_f128 : memref<3x3xf128>
  %output_f128Opening1 = memref.get_global @global_output_openinginter_f128 : memref<3x3xf128>
  %output_f128Closing = memref.get_global @global_output_closing_f128 : memref<3x3xf128>
  %output_f128Closing1 = memref.get_global @global_output_closinginter_f128 : memref<3x3xf128>
  %copyMemref128_1 = memref.get_global @global_copymemref1_f128 : memref<3x3xf128>
  %copyMemreff128_2 = memref.get_global @global_copymemref2_f128 : memref<3x3xf128>
  %c_f128 = arith.constant 0. : f128

  %kernelAnchorX = arith.constant 1 : index
  %kernelAnchorY = arith.constant 1 : index
  %iterations = arith.constant 1 : index

  dip.erosion_2d <CONSTANT_PADDING> %input_i32, %identity_f32, %output_f32Erosion, %copyMemreff32_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  // CHECK: 'dip.erosion_2d' op input, kernel, output, copymemref and constant must have the same element type

  dip.erosion_2d <CONSTANT_PADDING> %input_f32, %identity_i32, %output_f32Erosion, %copyMemreff32_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  // CHECK: 'dip.erosion_2d' op input, kernel, output, copymemref and constant must have the same element type

  dip.erosion_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_i32Erosion , %copyMemrefi32_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xi32>, memref<3x3xi32>, index, index,index, f32
  // CHECK: 'dip.erosion_2d' op input, kernel, output, copymemref and constant must have the same element type

  dip.erosion_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_f32Erosion, %copyMemreff32_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_i32: memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index,index, i32
  // CHECK: 'dip.erosion_2d' op input, kernel, output, copymemref and constant must have the same element type

  dip.erosion_2d <CONSTANT_PADDING> %input_f128, %identity_f128, %output_f128Erosion, %copyMemreff128_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f128 : memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, index, index,index, f128
  // CHECK: 'dip.erosion_2d' op supports only f32, f64 and integer types. 'f128'is passed

  dip.dilation_2d <CONSTANT_PADDING> %input_i32, %identity_f32, %output_f32Dilation, %copyMemreff32_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  // CHECK: 'dip.dilation_2d' op input, kernel, output, copymemref and constant must have the same element type

  dip.dilation_2d <CONSTANT_PADDING> %input_f32, %identity_i32, %output_f32Dilation,%copyMemreff32_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  // CHECK: 'dip.dilation_2d' op input, kernel, output, copymemref and constant must have the same element type

  dip.dilation_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_i32Dilation, %copyMemreff32_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, index, index,index, f32
  // CHECK: 'dip.dilation_2d' op input, kernel, output, copymemref and constant must have the same element type

  dip.dilation_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_f32Dilation, %copyMemreff32_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_i32 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index,index, i32
  // CHECK: 'dip.dilation_2d' op input, kernel, output, copymemref and constant must have the same element type

  dip.dilation_2d <CONSTANT_PADDING> %input_f128, %identity_f128, %output_f128Dilation, %copyMemref128_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f128 : memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, index, index,index, f128
  // CHECK: 'dip.dilation_2d' op supports only f32, f64 and integer types. 'f128'is passed

 dip.opening_2d <CONSTANT_PADDING> %input_i32, %identity_f32, %output_f32Opening, %output_f32Opening1, %copyMemreff32_2, %copyMemreff32_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  // CHECK: 'dip.opening_2d' op input, kernel, output, output1, copymemref, copymemref1 and constant must have the same element type

  dip.opening_2d <CONSTANT_PADDING> %input_f32, %identity_i32, %output_f32Opening, %output_f32Opening1, %copyMemreff32_2, %copyMemreff32_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  // CHECK: 'dip.opening_2d' op input, kernel, output, output1, copymemref, copymemref1 and constant must have the same element type

  dip.opening_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_i32Opening, %output_i32Opening1, %copyMemreff32_2, %copyMemreff32_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xi32>, memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, index, index,index, f32
  // CHECK: 'dip.opening_2d' op input, kernel, output, output1, copymemref, copymemref1 and constant must have the same element type

  dip.opening_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_f32Opening, %output_f32Opening1, %copyMemreff32_2, %copyMemreff32_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_i32 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index,index, i32
  // CHECK: 'dip.opening_2d' op input, kernel, output, output1, copymemref, copymemref1 and constant must have the same element type

  dip.opening_2d <CONSTANT_PADDING> %input_f128, %identity_f128, %output_f128Opening, %output_f128Opening1, %copyMemreff128_2, %copyMemref128_1, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f128 : memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, index, index,index, f128
  // CHECK: 'dip.opening_2d' op supports only f32, f64 and integer types. 'f128'is passed

  dip.closing_2d <CONSTANT_PADDING> %input_i32, %identity_f32, %output_f32Closing, %output_f32Closing1, %copyMemreff32_1, %copyMemreff32_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  // CHECK: 'dip.closing_2d' op input, kernel, output, output1, copymemref, copymemref1 and constant must have the same element type

  dip.closing_2d <CONSTANT_PADDING> %input_f32, %identity_i32, %output_f32Closing, %output_f32Closing1, %copyMemreff32_1, %copyMemreff32_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, index, f32
  // CHECK: 'dip.closing_2d' op input, kernel, output, output1, copymemref, copymemref1 and constant must have the same element type

  dip.closing_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_i32Closing, %output_i32Closing1, %copyMemrefi32_1, %copyMemrefi32_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f32 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>, index, index,index, f32
  // CHECK: 'dip.closing_2d' op input, kernel, output, output1, copymemref, copymemref1 and constant must have the same element type

  dip.closing_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_f32Closing, %output_f32Closing1, %copyMemreff32_1, %copyMemreff32_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_i32 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index,index, i32
  // CHECK: 'dip.closing_2d' op input, kernel, output, output1, copymemref, copymemref1 and constant must have the same element type

  dip.closing_2d <CONSTANT_PADDING> %input_f128, %identity_f128, %output_f128Closing, %output_f128Closing1, %copyMemref128_1, %copyMemreff128_2, %kernelAnchorX, %kernelAnchorY, %iterations, %c_f128 : memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, index, index,index, f128
  // CHECK: 'dip.closing_2d' op supports only f32, f64 and integer types. 'f128'is passed
  %ret = arith.constant 0 : i32
  return %ret : i32
}
