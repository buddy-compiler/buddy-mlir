// RUN: buddy-opt %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -convert-vector-to-scf \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 + d1)>

module {
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @pooling_nhwc_max(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %vl_step = arith.constant 32 : index
    %c0_f32 = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %c0_f32 : vector<32xf32>
    %dim = memref.dim %arg1, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg2, %c0 : memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
    %dim_3 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %dim_4 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>

    // Calculate the upper bound for vectorized processing
    // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
    // - Add 1 to ensure the final loop runs when the workload length
    //   is divisible by the vector size.
    %dim_4_upbound_tmp = arith.subi %dim_4, %vl_step : index
    %dim_4_upbound = arith.addi %dim_4_upbound_tmp, %c1 : index

    %t_start = call @rtclock() : () -> f64
    affine.for %arg3 = #map(%c0) to #map(%dim_1) {
      affine.for %arg4 = #map(%c0) to #map(%dim_2) {
        affine.for %arg5 = #map(%c0) to #map(%dim_3) {
          // Perform the vectorization body.
          %iter_idx = scf.for %arg6 = %c0 to %dim_4_upbound
              step %vl_step iter_args(%iter_init = %c0) -> (index) {      // N
            %4 = vector.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
            %5 = affine.for %arg7 = #map(%c0) to #map(%dim) iter_args(%arg8 = %4) -> (vector<32xf32>) {
              %6 = affine.for %arg9 = #map(%c0) to #map(%dim_0) iter_args(%arg10 = %arg8) -> (vector<32xf32>) {
                %in_iter_h = affine.apply #map1 (%arg7, %arg4)
                %in_iter_w = affine.apply #map1 (%arg9, %arg5)
                %7 = vector.load %arg0[%arg3, %in_iter_h, %in_iter_w, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
                %8 = arith.maximumf %7, %arg10 : vector<32xf32>
                affine.yield %8 : vector<32xf32>
              }
              affine.yield %6 : vector<32xf32>
            }
           vector.store %5, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
            %dim_4_next = arith.addi %dim_4, %vl_step : index
            scf.yield %dim_4_next : index
          }
          // Compute the tail size and Process the remaining elements
          // using masked vector operations.
          %tail_size = arith.subi %dim_4, %iter_idx : index
          %3 = arith.cmpi sgt, %tail_size, %c0 : index
          scf.if %3 {
            %mask = vector.create_mask %tail_size : vector<32xi1>
            %5 = vector.maskedload %arg2[%arg3, %arg4, %arg5, %iter_idx], %mask, %0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
            %6 = affine.for %arg7 = #map(%c0) to #map(%dim) iter_args(%arg8 = %5) -> (vector<32xf32>) {
              %8 = arith.addi %arg4, %arg7 : index
              %7 = affine.for %arg9 = #map(%c0) to #map(%dim_0) iter_args(%arg10 = %arg8) -> (vector<32xf32>) {
                %9 = arith.addi %arg9, %arg5 : index
                %10 = vector.maskedload %arg0[%arg3, %8, %9, %iter_idx], %mask, %0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
                %11 = arith.maximumf %10, %arg10 : vector<32xf32>
                affine.yield %11 : vector<32xf32>
              }
              affine.yield %7 : vector<32xf32>
            }
            vector.maskedstore %arg2[%arg3, %arg4, %arg5, %iter_idx], %mask, %6 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
          }
        }
      }
    }
    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    %printed_output = memref.cast %arg2 : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%printed_output) : (memref<*xf32>) -> ()

    // Print timings.
    vector.print %time : f64

    return
  }

  func.func @main(){
    // Set up dims.
    %c1 = arith.constant 1 : index
    %cInput = arith.constant 24 : index
    %cKernel = arith.constant 2 : index
    %cOutput = arith.constant 12 : index
    %c6 = arith.constant 6 : index

    // Set Init Value.
    %cf1_32 = arith.constant 1.0 : f32

    %a = memref.alloc(%c1, %cInput, %cInput, %c6) : memref<?x?x?x?xf32>
    %b = memref.alloc(%cKernel, %cKernel) : memref<?x?xf32>
    %c = memref.alloc(%c1, %cOutput, %cOutput, %c6) : memref<?x?x?x?xf32>

    linalg.fill ins(%cf1_32 : f32) outs(%a : memref<?x?x?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%b : memref<?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%c : memref<?x?x?x?xf32>)

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref
    // CHECK: [
    // CHECK: [
    // CHECK: [
    // CHECK: [1{{(, 1)*}}],
    call @pooling_nhwc_max(%a, %b, %c) : (memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>) -> ()

    memref.dealloc %c : memref<?x?x?x?xf32>
    memref.dealloc %b : memref<?x?xf32>
    memref.dealloc %a : memref<?x?x?x?xf32>

    return
  }
}
