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

func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>)

func.func @test(%arg0 : memref<?x?x?xf32>, %arg1 : memref<?x?x?xf32>, %arg2 : memref<?x?x?xf32>) {
  %t_start = call @rtclock() : () -> f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %vl_step = arith.constant 32 : index
  %c0_f32 = arith.constant 0.000000e+00 : f32
  %v0 = vector.splat %c0_f32 : vector<32xf32>
  %dim = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %dim_1 = memref.dim %arg0, %c2 : memref<?x?x?xf32>
  %dim_2 = memref.dim %arg1, %c1 : memref<?x?x?xf32>

  // Calculate the upper bound for vectorized processing
  // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
  // - Add 1 to ensure the final loop runs when the workload length
  //   is divisible by the vector size.
  %dim_1_upbound_tmp = arith.subi %dim_1, %vl_step : index
  %dim_1_upbound = arith.addi %dim_1_upbound_tmp, %c1 : index

  affine.for %arg3 = %c0 to %dim {
    affine.for %arg4 = %c0 to %dim_0 {
      affine.for %arg5 = %c0 to %dim_2 {
        %2 = affine.load %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>
        %iter_idx, %iter_value = scf.for %arg6 = %c0 to %dim_1_upbound
            step %vl_step iter_args(%iter_init = %c0, %iter_value0 = %2) -> (index, f32){
          %0 =vector.load %arg0[%arg3, %arg4, %arg6] : memref<?x?x?xf32>, vector<32xf32>
          %1 = vector.load %arg1[%arg3, %arg5, %arg6] : memref<?x?x?xf32>, vector<32xf32>
          %3 = arith.mulf %0, %1 : vector<32xf32>
          %4 = vector.reduction <add>, %3, %iter_value0 fastmath<reassoc> : vector<32xf32> into f32
          %dim_1_next = arith.addi %arg6, %vl_step : index
          scf.yield %dim_1_next, %4 : index, f32
        }
        // Compute the tail size and Process the remaining elements
        // using masked vector operations.
        %tail_size = arith.subi %dim_1, %iter_idx : index
        %mask = vector.create_mask %tail_size : vector<32xi1>
        %0 = vector.maskedload %arg1[%arg3, %arg4, %iter_idx], %mask, %v0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        %10 = vector.maskedload %arg1[%arg3, %arg5, %iter_idx], %mask, %v0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        %3 = arith.mulf %0, %10 : vector<32xf32>
        %4 = vector.reduction <add>, %3, %iter_value fastmath<reassoc> : vector<32xf32> into f32
        affine.store %4, %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>
      }
    }
  }
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  // Print timings.
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
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32

  %m0 = call @alloc_f32(%c32, %c64, %c64, %f1) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m1 = call @alloc_f32(%c32, %c64, %c64, %f1) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m2 = call @alloc_f32(%c32, %c64, %c64, %f0) : (index, index, index, f32) -> memref<?x?x?xf32>

  call @test(%m0, %m1, %m2) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()

  %printed_m2 = memref.cast %m2 : memref<?x?x?xf32> to memref<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [32, 64, 64] strides = [4096, 64, 1] data =
  // CHECK-NEXT: [
  // CHECK: [
  // CHECK: [64{{(, 64)*}}]
  call @printMemrefF32(%printed_m2) : (memref<*xf32>) -> ()

  %m3 = call @alloc_f32(%c3, %c3, %c32, %f1) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m4 = call @alloc_f32(%c3, %c3, %c32, %f1) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m5 = call @alloc_f32(%c3, %c3, %c3, %f0) : (index, index, index, f32) -> memref<?x?x?xf32>

  call @test(%m3, %m4, %m5) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()

  %printed_m5 = memref.cast %m5 : memref<?x?x?xf32> to memref<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [3, 3, 3] strides = [9, 3, 1] data =
  // CHECK-NEXT: [
  // CHECK: [
  // CHECK: [32{{(, 32)*}}]
  call @printMemrefF32(%printed_m5) : (memref<*xf32>) -> ()

  return
}
