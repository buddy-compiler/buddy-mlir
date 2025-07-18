// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -expand-strided-metadata \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64

  // CMK * CKN -> CMN
  func.func @batch_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %vl_step = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %cst : vector<32xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x?x?xf32>
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
    %dim_2 = memref.dim %arg1, %c1 : memref<?x?x?xf32>
    %dim_3 = memref.dim %arg1, %c2 : memref<?x?x?xf32>

    // Calculate the upper bound for vectorized processing
    // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
    // - Add 1 to ensure the final loop runs when the workload length
    //   is divisible by the vector size.
    %dim_3_upbound_tmp = arith.subi %dim_3, %vl_step : index
    %dim_3_upbound = arith.addi %dim_3_upbound_tmp, %c1 : index

    %t_start = call @rtclock() : () -> f64
    affine.for %arg3 = %c0 to %dim {                                      // C
      affine.prefetch %arg0[%arg3, %dim_1, %dim_2], read, locality<3>, data : memref<?x?x?xf32>
      affine.for %arg4 = %c0 to %dim_1 {                                  // M
        // Perform the vectorization body.
        %iter_idx = scf.for %arg5 = %c0 to %dim_3_upbound
              step %vl_step iter_args(%iter_init = %c0) -> (index) {      // N
          %1 = vector.load %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>, vector<32xf32>
          %iter_vec = scf.for %arg6 = %c0 to %dim_2 step %c1
              iter_args(%iter_vec0 = %1) -> (vector<32xf32>) {            // K
            %5 = memref.load %arg0[%arg3, %arg4, %arg6] : memref<?x?x?xf32>
            %6 = vector.broadcast %5 : f32 to vector<32xf32>
            %4 = vector.load %arg1[%arg3, %arg6, %arg5] : memref<?x?x?xf32>, vector<32xf32>
            %8 = vector.fma %6, %4, %iter_vec0  : vector<32xf32>
            scf.yield %8 : vector<32xf32>
          }
          vector.store %iter_vec, %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>, vector<32xf32>
          %arg5_next = arith.addi %arg5, %vl_step : index
          scf.yield %arg5_next : index
        }
        // Compute the tail size and Process the remaining elements
        // using masked vector operations.
        %tail_size = arith.subi %dim_3, %iter_idx : index
        %3 = arith.cmpi sgt, %tail_size, %c0 : index
        scf.if %3 {
          %mask = vector.create_mask %tail_size : vector<32xi1>
          %1 = vector.maskedload %arg2[%arg3, %arg4, %iter_idx], %mask, %0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %iter_vec = scf.for %arg6 = %c0 to %dim_2 step %c1
              iter_args(%iter_vec0 = %1) -> (vector<32xf32>) {             // K
            %5 = vector.maskedload %arg1[%arg3, %arg6, %iter_idx], %mask, %0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
            %6 = memref.load %arg0[%arg3, %arg4, %arg6] : memref<?x?x?xf32>
            %7 = vector.broadcast %6 : f32 to vector<32xf32>
            %9 = vector.fma %7, %5, %iter_vec0 : vector<32xf32>
            scf.yield %9 : vector<32xf32>
          }
          vector.maskedstore %arg2[%arg3, %arg4, %iter_idx], %mask, %iter_vec : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32>
        }
      }
    }
    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    %printed_output = memref.cast %arg2 : memref<?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%printed_output) : (memref<*xf32>) -> ()
    vector.print %time : f64
    return
  }
  func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg4: f32) -> memref<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2) : memref<?x?x?xf32>
    scf.for %idx0 = %c0 to %arg0 step %c1 {
      scf.for %idx1 = %c0 to %arg1 step %c1 {
        scf.for %idx2 = %c0 to %arg2 step %c1 {
          memref.store %arg4, %0[%idx0, %idx1, %idx2] : memref<?x?x?xf32>
        }
      }
    }
    return %0 : memref<?x?x?xf32>
  }

  func.func @main(){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c576 = arith.constant 576 : index
    %c1024 = arith.constant 1024 : index
    %c1000 = arith.constant 1000 : index
    %f0 = arith.constant 0.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f3 = arith.constant 3.0 : f32

    %m0 = call @alloc_f32(%c1, %c1, %c576, %f2) : (index, index, index, f32) -> memref<?x?x?xf32>
    %m1 = call @alloc_f32(%c1, %c576, %c1024, %f3) : (index, index, index, f32) -> memref<?x?x?xf32>
    %m2 = call @alloc_f32(%c1, %c1, %c1024, %f0) : (index, index, index, f32) -> memref<?x?x?xf32>

    // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 1, 1024] strides = [1024, 1024, 1] data =
    // CHECK-NEXT: [
    // CHECK: [
    // CHECK: [3456{{(, 3456)*}}]
    call @batch_matmul(%m0, %m1, %m2) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()

    %m3 = call @alloc_f32(%c1, %c1, %c1024, %f2) : (index, index, index, f32) -> memref<?x?x?xf32>
    %m4 = call @alloc_f32(%c1, %c1024, %c1000, %f3) : (index, index, index, f32) -> memref<?x?x?xf32>
    %m5 = call @alloc_f32(%c1, %c1, %c1000, %f0) : (index, index, index, f32) -> memref<?x?x?xf32>

    // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 1, 1000] strides = [1000, 1000, 1] data =
    // CHECK-NEXT: [
    // CHECK: [
    // CHECK: [6144{{(, 6144)*}}]
    call @batch_matmul(%m3, %m4, %m5) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()

    return
  }
}
