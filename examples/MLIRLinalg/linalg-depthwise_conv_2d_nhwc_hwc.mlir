// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    scf.for %arg5 = %c0 to %arg0 step %c1 {
      scf.for %arg6 = %c0 to %arg1 step %c1 {
        scf.for %arg7 = %c0 to %arg2 step %c1 {
          scf.for %arg8 = %c0 to %arg3 step %c1 {
            %iarg8 = arith.index_cast %arg8 : index to i32
            %loopf = arith.sitofp %iarg8 : i32 to f32
            memref.store %loopf, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }

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

    %image_n = arith.constant 2 : index
    %image_h = arith.constant 8 : index
    %image_w = arith.constant 8 : index
    %image_c = arith.constant 18 : index

    %filter_h = arith.constant 4 : index
    %filter_w = arith.constant 4 : index
    %filter_c = arith.constant 18 : index

    %output_n = arith.constant 2 : index
    %output_h = arith.constant 5 : index
    %output_w = arith.constant 5 : index
    %output_c = arith.constant 18 : index

    // Allocate and fill image, filter, and output.
    %image = call @alloc_2d_filled_f32(%image_n, %image_h, %image_w, %image_c, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %filter = call @alloc_2d_filled_f32(%filter_h, %filter_w, %filter_c, %cst) : (index, index, index, f32) -> memref<?x?x?xf32>
    %output = call @alloc_2d_filled_f32(%output_n, %output_h, %output_w, %output_c, %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

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
