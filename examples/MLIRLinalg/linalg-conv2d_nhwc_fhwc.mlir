// RUN: buddy-opt %s \
// RUN:     -conv-nhwc-fhwc-optimize -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
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
            %iarg8=arith.index_cast %arg8 : index to i32
            %loopf= arith.sitofp %iarg8 : i32 to f32
            memref.store %loopf, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }
  func.func @conv_2d_nhwc_fhwc(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    linalg.conv_2d_nhwc_fhwc ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?xf32>)
    return
  }
  func.func @main() {
    // Intput(image, filter) and output value.
    %cst = arith.constant 0.500000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32

    %current_image_n = arith.constant 1 : index
    %current_image_c = arith.constant 2 : index
    %current_image_h = arith.constant 4 : index
    %current_image_w = arith.constant 4 : index

    %current_filter_f = arith.constant 2 : index
    %current_filter_c = arith.constant 2 : index
    %current_filter_h = arith.constant 2 : index
    %current_filter_w = arith.constant 2 : index

    %current_output_n = arith.constant 1 : index
    %current_output_c = arith.constant 2 : index
    %current_output_h = arith.constant 3 : index
    %current_output_w = arith.constant 3 : index

    // Image.
    %image = call @alloc_2d_filled_f32(%current_image_n,%current_image_h, %current_image_w, %current_image_c,  %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    // Filter.
    %filter = call @alloc_2d_filled_f32(%current_filter_f, %current_filter_h, %current_filter_w,%current_filter_c,  %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    // Output.
    %output = call @alloc_2d_filled_f32(%current_output_n, %current_output_h, %current_output_w,%current_output_c,  %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    call @conv_2d_nhwc_fhwc(%image, %filter, %output) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

    %3 = memref.cast %output : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%3) : (memref<*xf32>) -> ()


    memref.dealloc %output : memref<?x?x?x?xf32>
    memref.dealloc %image : memref<?x?x?x?xf32>
    memref.dealloc %filter : memref<?x?x?x?xf32>
    return
  }
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 3, 3, 2] strides = [18, 6, 2, 1] data =
// CHECK{LITERAL}: [[[[4,     5],
// CHECK{LITERAL}:    [4,     5],
// CHECK{LITERAL}:    [4,     5]],
// CHECK{LITERAL}:   [[4,     5],
// CHECK{LITERAL}:    [4,     5],
// CHECK{LITERAL}:    [4,     5]],
// CHECK{LITERAL}:   [[4,     5],
// CHECK{LITERAL}:    [4,     5],
// CHECK{LITERAL}:    [4,     5]]]]
