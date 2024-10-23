// RUN: buddy-opt %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -convert-vector-to-scf \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 ceildiv 32)>

module {
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @pooling_nhwc_max(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index
    %c0_f32 = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %c0_f32 : vector<32xf32>
    %dim = memref.dim %arg1, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg2, %c0 : memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
    %dim_3 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %dim_4 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>
    affine.for %arg3 = #map(%c0) to #map(%dim_1) {
      affine.for %arg4 = #map(%c0) to #map(%dim_2) {
        affine.for %arg5 = #map(%c0) to #map(%dim_3) {
          affine.for %arg6 = #map(%c0) to #map1(%dim_4) {
            %1 = arith.muli %arg6, %c32 : index
            %2 = arith.subi %dim_4, %1 : index
            %3 = arith.cmpi sge, %2, %c32 : index
            scf.if %3 {
              %4 = affine.vector_load %arg2[%arg3, %arg4, %arg5, %arg6 * 32] : memref<?x?x?x?xf32>, vector<32xf32>
              %5 = affine.for %arg7 = #map(%c0) to #map(%dim) iter_args(%arg8 = %4) -> (vector<32xf32>) {
                %6 = affine.for %arg9 = #map(%c0) to #map(%dim_0) iter_args(%arg10 = %arg8) -> (vector<32xf32>) {
                  %7 = affine.vector_load %arg0[%arg3, %arg7 + %arg4, %arg9 + %arg5, %arg6 * 32] : memref<?x?x?x?xf32>, vector<32xf32>
                  %8 = arith.maximumf %7, %arg10 : vector<32xf32>
                  affine.yield %8 : vector<32xf32>
                }
                affine.yield %6 : vector<32xf32>
              }
              affine.vector_store %5, %arg2[%arg3, %arg4, %arg5, %arg6 * 32] : memref<?x?x?x?xf32>, vector<32xf32>
            } else {
              %4 = vector.create_mask %2 : vector<32xi1>
              %5 = vector.maskedload %arg2[%arg3, %arg4, %arg5, %1], %4, %0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
              %6 = affine.for %arg7 = #map(%c0) to #map(%dim) iter_args(%arg8 = %5) -> (vector<32xf32>) {
                %8 = arith.addi %arg4, %arg7 : index
                %7 = affine.for %arg9 = #map(%c0) to #map(%dim_0) iter_args(%arg10 = %arg8) -> (vector<32xf32>) {
                  %9 = arith.addi %arg9, %arg5 : index
                  %10 = vector.maskedload %arg0[%arg3, %8, %9, %1], %4, %0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
                  %11 = arith.maximumf %10, %arg10 : vector<32xf32>
                  affine.yield %11 : vector<32xf32>
                }
                affine.yield %7 : vector<32xf32>
              }
              vector.maskedstore %arg2[%arg3, %arg4, %arg5, %1], %4, %6 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
            }
          }
        }
      }
    }
    return
  }

  func.func @main(){
    // Set up dims.
    %c1 = arith.constant 1 : index
    %cInput = arith.constant 24 : index
    %cKernel = arith.constant 2 : index
    %cOutput = arith.constant 23 : index
    %c6 = arith.constant 6 : index

    // Set Init Value.
    %cf1_32 = arith.constant 1.0 : f32

    %a = memref.alloc(%c1, %cInput, %cInput, %c6) : memref<?x?x?x?xf32>
    %b = memref.alloc(%cKernel, %cKernel) : memref<?x?xf32>
    %c = memref.alloc(%c1, %cOutput, %cOutput, %c6) : memref<?x?x?x?xf32>

    linalg.fill ins(%cf1_32 : f32) outs(%a : memref<?x?x?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%b : memref<?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%c : memref<?x?x?x?xf32>)

    %t0 = call @rtclock() : () -> f64
    call @pooling_nhwc_max(%a, %b, %c) : (memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>) -> ()
    %t1 = call @rtclock() : () -> f64
    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref
    // CHECK: [
    // CHECK: [
    // CHECK: [
    // CHECK: [1{{(, 1)*}}],
    %print_C = memref.cast %c : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()
    %time = arith.subf %t1, %t0 : f64
    vector.print %time : f64

    memref.dealloc %c : memref<?x?x?x?xf32>
    memref.dealloc %b : memref<?x?xf32>
    memref.dealloc %a : memref<?x?x?x?xf32>

    return 
  }
}
