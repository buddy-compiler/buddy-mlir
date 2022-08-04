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

memref.global "private" @global_output_f32 : memref<3x3xf32> = dense<[[0., 0., 0.],
                                                                      [0., 0., 0.],
                                                                      [0., 0., 0.]]>

memref.global "private" @global_input_i32 : memref<3x3xi32> = dense<[[0 , 1 , 2 ],
                                                                     [10, 11, 12],
                                                                     [20, 21, 22]]>

memref.global "private" @global_identity_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                        [0, 1, 0],
                                                                        [0, 0, 0]]>

memref.global "private" @global_output_i32 : memref<3x3xi32> = dense<[[0, 0, 0],
                                                                      [0, 0, 0],
                                                                      [0, 0, 0]]>

memref.global "private" @global_input_f128 : memref<3x3xf128> = dense<[[0. , 1. , 2. ],
                                                                       [10., 11., 12.],
                                                                       [20., 21., 22.]]>

memref.global "private" @global_output_f128 : memref<3x3xf128> = dense<[[0., 0., 0. ],
                                                                        [0., 1., 0.],
                                                                        [0., 0., 0.]]>

memref.global "private" @global_identity_f128 : memref<3x3xf128> = dense<[[0., 0., 0.],
                                                                          [0., 0., 0.],
                                                                          [0., 0., 0.]]>

func.func @main() -> i32 {
  %input_f32 = memref.get_global @global_input_f32 : memref<3x3xf32>
  %identity_f32 = memref.get_global @global_identity_f32 : memref<3x3xf32>
  %output_f32 = memref.get_global @global_output_f32 : memref<3x3xf32>
  %c_f32 = arith.constant 0. : f32 

  %input_i32 = memref.get_global @global_input_i32 : memref<3x3xi32>
  %identity_i32 = memref.get_global @global_identity_i32 : memref<3x3xi32>
  %output_i32 = memref.get_global @global_output_i32 : memref<3x3xi32>
  %c_i32 = arith.constant 0 : i32 

  %input_f128 = memref.get_global @global_input_f128 : memref<3x3xf128>
  %identity_f128 = memref.get_global @global_identity_f128 : memref<3x3xf128>
  %output_f128 = memref.get_global @global_output_f128 : memref<3x3xf128>
  %c_f128 = arith.constant 0. : f128

  %kernelAnchorX = arith.constant 1 : index
  %kernelAnchorY = arith.constant 1 : index

  dip.corr_2d <CONSTANT_PADDING> %input_i32, %identity_f32, %output_f32, %kernelAnchorX, %kernelAnchorY, %c_f32 : memref<3x3xi32>, memref<3x3xf32>, memref<3x3xf32>, index, index, f32
  // CHECK: 'dip.corr_2d' op input, kernel, output and constant must have the same element type

  dip.corr_2d <CONSTANT_PADDING> %input_f32, %identity_i32, %output_f32, %kernelAnchorX, %kernelAnchorY, %c_f32 : memref<3x3xf32>, memref<3x3xi32>, memref<3x3xf32>, index, index, f32
  // CHECK: 'dip.corr_2d' op input, kernel, output and constant must have the same element type

  dip.corr_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_i32, %kernelAnchorX, %kernelAnchorY, %c_f32 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xi32>, index, index, f32
  // CHECK: 'dip.corr_2d' op input, kernel, output and constant must have the same element type

  dip.corr_2d <CONSTANT_PADDING> %input_f32, %identity_f32, %output_f32, %kernelAnchorX, %kernelAnchorY, %c_i32 : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, index, index, i32
  // CHECK: 'dip.corr_2d' op input, kernel, output and constant must have the same element type

  dip.corr_2d <CONSTANT_PADDING> %input_f128, %identity_f128, %output_f128, %kernelAnchorX, %kernelAnchorY, %c_f128 : memref<3x3xf128>, memref<3x3xf128>, memref<3x3xf128>, index, index, f128
  // CHECK: 'dip.corr_2d' op supports only f32, f64 and integer types. 'f128'is passed
  %ret = arith.constant 0 : i32
  return %ret : i32
}
