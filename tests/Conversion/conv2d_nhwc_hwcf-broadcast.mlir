// RUN: buddy-opt %s \
// RUN:     -conv-broadcast="stride=32" \
// RUN:     -convert-linalg-to-loops -convert-vector-to-scf -lower-affine \
// RUN:     -convert-scf-to-cf -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm -convert-cf-to-llvm -reconcile-unrealized-cas \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_lib \
// RUN:     -shared-libs=%mlir_c_runner_utils_lib \
// RUN: | FileCheck %s
module {
    func.func private @printMemrefF32(memref<*xf32>)
    func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %val: f32) -> memref<?x?x?x?xf32> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
        scf.for %arg5 = %c0 to %arg0 step %c1 {
            scf.for %arg6 = %c0 to %arg1 step %c1 {
                scf.for %arg7 = %c0 to %arg2 step %c1 {
                    scf.for %arg8 = %c0 to %arg3 step %c1 {
                        memref.store %val, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
                    }
                }
            }
        }
        return %0 : memref<?x?x?x?xf32>
    }

    func.func @conv_2d_nhwc_hwcf(%a: memref<?x?x?x?xf32>, %b: memref<?x?x?x?xf32>, %c: memref<?x?x?x?xf32>) {
        linalg.conv_2d_nhwc_hwcf
            ins(%a, %b: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
            outs(%c: memref<?x?x?x?xf32>)
        return
    }

    func.func @main() {
        // Input and kernel value.
        %cst = arith.constant 1.000000e+00 : f32
        // Output value.
        %cst_0 = arith.constant 0.000000e+00 : f32
        
        // Define layout.
        %input_n = arith.constant 1 : index
        %input_h = arith.constant 3 : index
        %input_w = arith.constant 3 : index
        %input_c = arith.constant 2 : index
        
        %kernel_h = arith.constant 2 : index
        %kernel_w = arith.constant 2 : index
        %kernel_c = arith.constant 2 : index
        %kernel_f = arith.constant 2 : index
        
        %output_n = arith.constant 1 : index
        %output_h = arith.constant 2 : index
        %output_w = arith.constant 2 : index
        %output_c = arith.constant 2 : index
        
        
        // Define input, kernel, and output memref.
        %input = call @alloc_f32(%input_n, %input_h, %input_w, %input_c, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
        %kernel = call @alloc_f32(%kernel_h, %kernel_w, %kernel_c, %kernel_f, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
        %output = call @alloc_f32(%output_n, %output_h, %output_w, %output_c, %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

        // Perform convolution
        call @conv_2d_nhwc_hwcf(%input, %kernel, %output) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

        // Print the output
        // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 2, 2, 2] strides = [8, 4, 2, 1] data =
        // CHECK-NEXT: [
        // CHECK-SAME:  [
        // CHECK-SAME:   [
        // CHECK-SAME:    [8,     8],
        // CHECK-NEXT:    [8,     8]
        // CHECK-SAME:   ],
        // CHECK-NEXT:   [
        // CHECK-SAME:    [8,     8],
        // CHECK-NEXT:    [8,     8]
        // CHECK-SAME:   ]
        // CHECK-SAME:  ]
        // CHECK-SAME: ]
        %print_output = memref.cast %output : memref<?x?x?x?xf32> to memref<*xf32>
        call @printMemrefF32(%print_output) : (memref<*xf32>) -> ()

        memref.dealloc %output : memref<?x?x?x?xf32>
        memref.dealloc %input : memref<?x?x?x?xf32>
        memref.dealloc %kernel : memref<?x?x?x?xf32>
        return
    }
}