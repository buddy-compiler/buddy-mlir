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

    func.func @matmul(%a : memref<?x?xf16>, %b : memref<?x?xf16>, %c : memref<?x?xf16>) {
        linalg.matmul
            ins(%a, %b: memref<?x?xf16>, memref<?x?xf16>)
            outs(%c: memref<?x?xf16>)
        return
    }

    func.func @main() {
        // Set up dims.
        %cM = arith.constant 2 : index
        %cN = arith.constant 2 : index
        %cK = arith.constant 2 : index

        // Set Init Value.
        %cf2048 = arith.constant 2048.000000e+00 : f16
        %cf2049 = arith.constant 2049.000000e+00 : f16
        %cf0 = arith.constant 0.000000e+00 : f16
        %cf2 = arith.constant 2.000000e+00 : f16

        // Allocate and initialize matrices
        %A1 = call @alloc_2d_filled_f16(%cM, %cK, %cf2048) : (index, index, f16) -> memref<?x?xf16>
        %A2 = call @alloc_2d_filled_f16(%cM, %cK, %cf2049) : (index, index, f16) -> memref<?x?xf16>
        %B = call @alloc_2d_filled_f16(%cK, %cN, %cf2) : (index, index, f16) -> memref<?x?xf16>
        %C1 = call @alloc_2d_filled_f16(%cM, %cN, %cf0) : (index, index, f16) -> memref<?x?xf16>
        %C2 = call @alloc_2d_filled_f16(%cM, %cN, %cf0) : (index, index, f16) -> memref<?x?xf16>

        call @matmul(%A1, %B, %C1) : (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>) -> ()
        call @matmul(%A2, %B, %C2) : (memref<?x?xf16>, memref<?x?xf16>, memref<?x?xf16>) -> ()

        // Convert f16 output to f32 for printing
        %C1_f32 = memref.alloc(%cM, %cN) : memref<?x?xf32>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %cM step %c1 {
            scf.for %j = %c0 to %cN step %c1 {
                %val_f16 = memref.load %C1[%i, %j] : memref<?x?xf16>
                %val_f32 = arith.extf %val_f16 : f16 to f32
                memref.store %val_f32, %C1_f32[%i, %j] : memref<?x?xf32>
            }
        }

        %C2_f32 = memref.alloc(%cM, %cN) : memref<?x?xf32>
        scf.for %i = %c0 to %cM step %c1 {
            scf.for %j = %c0 to %cN step %c1 {
                %val_f16 = memref.load %C2[%i, %j] : memref<?x?xf16>
                %val_f32 = arith.extf %val_f16 : f16 to f32
                memref.store %val_f32, %C2_f32[%i, %j] : memref<?x?xf32>
            }
        }

        // Print output.
        %print_C1 = memref.cast %C1_f32 : memref<?x?xf32> to memref<*xf32>
        %print_C2 = memref.cast %C2_f32 : memref<?x?xf32> to memref<*xf32>
        call @printMemrefF32(%print_C1) : (memref<*xf32>) -> ()
        call @printMemrefF32(%print_C2) : (memref<*xf32>) -> ()

        // Deallocations
        memref.dealloc %A1 : memref<?x?xf16>
        memref.dealloc %A2 : memref<?x?xf16>
        memref.dealloc %B : memref<?x?xf16>
        memref.dealloc %C1 : memref<?x?xf16>
        memref.dealloc %C2 : memref<?x?xf16>
        memref.dealloc %C1_f32 : memref<?x?xf32>
        memref.dealloc %C2_f32 : memref<?x?xf32>
        return
    }
}
