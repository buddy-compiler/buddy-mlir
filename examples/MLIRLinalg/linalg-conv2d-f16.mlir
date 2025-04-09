// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @alloc_2d_filled_f16(%arg0: index, %arg1: index, %arg2: f16) -> memref<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf16>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %arg2, %0[%arg3, %arg4] : memref<?x?xf16>
      }
    }
    return %0 : memref<?x?xf16>
  }

  func.func @conv_2d(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16>) {
    linalg.conv_2d ins (%arg0, %arg1: memref<?x?xf16>, memref<?x?xf16>)
                  outs (%arg2: memref<?x?xf16>)
    return
  }

  func.func @main() {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Image and Output value.
    %cst = arith.constant 1.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_2048 = arith.constant 2048.000000e+00 : f16
    %cst_2049 = arith.constant 2049.000000e+00 : f16

    %current_filter = arith.constant 3 : index
    %current_output = arith.constant 8 : index
    %current_image = affine.apply #map0(%current_output, %current_filter)

    // Filter.
    %filter = call @alloc_2d_filled_f16(%current_filter, %current_filter, %cst) : (index, index, f16) -> memref<?x?xf16>
    // Image.
    %image1 = call @alloc_2d_filled_f16(%current_image, %current_image, %cst_2048) : (index, index, f16) -> memref<?x?xf16>

    %image2 = call @alloc_2d_filled_f16(%current_image, %current_image, %cst_2049) : (index, index, f16) -> memref<?x?xf16>
    // Output.
    %output1 = call @alloc_2d_filled_f16(%current_output, %current_output, %cst_0) : (index, index, f16) -> memref<?x?xf16>

    %output2 = call @alloc_2d_filled_f16(%current_output, %current_output, %cst_0) : (index, index, f16) -> memref<?x?xf16>

    call @conv_2d(%image1, %filter, %output1) : (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>) -> ()

    call @conv_2d(%image2, %filter, %output2) : (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>) -> ()

    // Convert f16 output to f32 for printing.
    %output_f32_1 = memref.alloc(%current_output, %current_output) : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %current_output step %c1 {
      scf.for %j = %c0 to %current_output step %c1 {
        %val_f16 = memref.load %output1[%i, %j] : memref<?x?xf16>
        %val_f32 = arith.extf %val_f16 : f16 to f32
        memref.store %val_f32, %output_f32_1[%i, %j] : memref<?x?xf32>
      }
    }
    %print_output1 = memref.cast %output_f32_1 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_output1) : (memref<*xf32>) -> ()

    %output_f32_2 = memref.alloc(%current_output, %current_output) : memref<?x?xf32>
    scf.for %i = %c0 to %current_output step %c1 {
      scf.for %j = %c0 to %current_output step %c1 {
        %val_f16 = memref.load %output2[%i, %j] : memref<?x?xf16>
        %val_f32 = arith.extf %val_f16 : f16 to f32
        memref.store %val_f32, %output_f32_2[%i, %j] : memref<?x?xf32>
      }
    }
    %print_output2 = memref.cast %output_f32_2 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_output2) : (memref<*xf32>) -> ()

    memref.dealloc %image1 : memref<?x?xf16>
    memref.dealloc %image2 : memref<?x?xf16>
    memref.dealloc %filter : memref<?x?xf16>
    memref.dealloc %output1 : memref<?x?xf16>
    memref.dealloc %output2 : memref<?x?xf16>
    memref.dealloc %output_f32_1 : memref<?x?xf32>
    memref.dealloc %output_f32_2 : memref<?x?xf32>
    return
  }
}
