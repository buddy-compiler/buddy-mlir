// RUN: buddy-opt %s \
// RUN:     -depthwise-conv-nhwc-hwc-optimize -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @depthwise_conv_2d_nhwc_hwc(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<[1,1]> : tensor<2xi64>, strides = dense<[1,1]> : tensor<2xi64>}
      ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?xf32>)
      outs(%arg2 : memref<?x?x?x?xf32>)
    return
  }

  func.func @main() {
    // Constants for input image, filter, and output sizes.
    %cst = arith.constant 0.500000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cf1 = arith.constant 1.0 : f32

    %image_n = arith.constant 1 : index
    %image_h = arith.constant 4 : index
    %image_w = arith.constant 4 : index
    %image_c = arith.constant 2 : index

    %filter_h = arith.constant 1 : index
    %filter_w = arith.constant 2 : index
    %filter_c = arith.constant 2 : index

    %output_n = arith.constant 1 : index
    %output_h = arith.constant 3 : index
    %output_w = arith.constant 3 : index
    %output_c = arith.constant 2 : index

    %image = memref.alloc(%image_n,%image_h,%image_w,%image_c) : memref<?x?x?x?xf32>
    %filter = memref.alloc(%filter_h,%filter_w,%filter_c) : memref<?x?x?xf32>
    %output = memref.alloc(%output_n,%output_h,%output_w,%output_c) : memref<?x?x?x?xf32>

    // Allocate and fill image, filter, and output.
    linalg.fill
      ins(%cf1 : f32)
      outs(%image:memref<?x?x?x?xf32>)

    linalg.fill
      ins(%cf1 : f32)
      outs(%filter:memref<?x?x?xf32>)
    linalg.fill
      ins(%cf1 : f32)
      outs(%output:memref<?x?x?x?xf32>)

    // Call depthwise convolution.
    call @depthwise_conv_2d_nhwc_hwc(%image, %filter, %output) : (memref<?x?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

    %output_cast = memref.cast %output : memref<?x?x?x?xf32> to memref<*xf32>

    // Print the output.
    call @printMemrefF32(%output_cast) : (memref<*xf32>) -> ()

    // Deallocate memory.
    memref.dealloc %output : memref<?x?x?x?xf32>
    memref.dealloc %image : memref<?x?x?x?xf32>
    memref.dealloc %filter : memref<?x?x?xf32>
    return
  }
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 3, 3, 2] strides = [18, 6, 2, 1] data =
// CHECK{LITERAL}: [[[[3,     3],
// CHECK{LITERAL}:        [3,     3],
// CHECK{LITERAL}:        [3,     3]],
// CHECK{LITERAL}:       [[3,     3],
// CHECK{LITERAL}:        [3,     3],
// CHECK{LITERAL}:        [3,     3]],
// CHECK{LITERAL}:       [[3,     3],
// CHECK{LITERAL}:        [3,     3],
// CHECK{LITERAL}:        [3,     3]]]]
