// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -expand-strided-metadata \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>)

func.func @test(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>, %arg2 : memref<?x?x?x?xf32>) {
  %t_start = call @rtclock() : () -> f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %vl_step = arith.constant 32 : index
  %c0_f32 = arith.constant 0.000000e+00 : f32
  %v0 = vector.splat %c0_f32 : vector<32xf32>
  %dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
  %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
  %dim_1 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
  %dim_2 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>

  // Calculate the upper bound for vectorized processing
  // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
  // - Add 1 to ensure the final loop runs when the workload length 
  //   is divisible by the vector size.
  %dim_2_upbound_tmp = arith.subi %dim_2, %vl_step : index
  %dim_2_upbound = arith.addi %dim_2_upbound_tmp, %c1 : index

  affine.for %arg3 = %c0 to %dim {
    affine.for %arg4 = %c0 to %dim_1 {
      affine.for %arg5 = %c0 to %dim_0 {
        %iter_idx= scf.for %arg6 = %c0 to %dim_2_upbound 
            step %vl_step iter_args(%iter_init = %c0) -> (index){
          %0 = vector.load %arg0[%arg3, %arg5, %arg4, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
          %1 = vector.load %arg1[%arg3, %c0, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
          %2 = arith.mulf %0, %1 : vector<32xf32>
          vector.store %2, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<32xf32>
          %dim_2_next = arith.addi %arg6, %vl_step : index
          scf.yield %dim_2_next : index
        }
        // Compute the tail size and Process the remaining elements 
        // using masked vector operations.
        %tail_size = arith.subi %dim_2, %iter_idx : index
        %mask = vector.create_mask %tail_size : vector<32xi1>
        %0 = vector.maskedload %arg0[%arg3, %arg5, %arg4, %iter_idx], %mask, %v0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        %1 = vector.maskedload %arg1[%arg3, %c0, %arg5, %iter_idx], %mask, %v0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        %2 = arith.mulf %0, %1 : vector<32xf32>
        vector.maskedstore %arg2[%arg3, %arg4, %arg5, %iter_idx], %mask, %2 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
      }
    }
  }
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  // Print timings.
  vector.print %time : f64
  return
}

func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
  scf.for %idx0 = %c0 to %arg0 step %c1 {
    scf.for %idx1 = %c0 to %arg1 step %c1 {
      scf.for %idx2 = %c0 to %arg2 step %c1 {
        scf.for %idx3 = %c0 to %arg3 step %c1 {
          memref.store %arg4, %0[%idx0, %idx1, %idx2, %idx3] : memref<?x?x?x?xf32>
        }
      }
    }
  }
  return %0 : memref<?x?x?x?xf32>
}

func.func @main(){
  %c32 = arith.constant 32 : index
  %c40 = arith.constant 40 : index
  %c64 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : f32
  %f2 = arith.constant 2.0 : f32

  %m0 = call @alloc_f32(%c1, %c40, %c32, %c64, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  %m1 = call @alloc_f32(%c1, %c1, %c40, %c64, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  %m2 = call @alloc_f32(%c1, %c32, %c40, %c64, %f0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

  call @test(%m0, %m1, %m2) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

  %printed_m2 = memref.cast %m2 : memref<?x?x?x?xf32> to memref<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 32, 40, 128] strides = [163840, 5120, 128, 1] data = 
  // CHECK-NEXT: [
  // CHECK: [
  // CHECK: [
  // CHECK: [4{{(, 4)*}}]
  call @printMemrefF32(%printed_m2) : (memref<*xf32>) -> ()

  %m3 = call @alloc_f32(%c1, %c4, %c3, %c32, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  %m4 = call @alloc_f32(%c1, %c1, %c4, %c32, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  %m5 = call @alloc_f32(%c1, %c3, %c4, %c32, %f0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

  call @test(%m3, %m4, %m5) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

  %printed_m5 = memref.cast %m5 : memref<?x?x?x?xf32> to memref<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 3, 4, 32] strides = [384, 128, 32, 1] data =
  // CHECK-NEXT: [
  // CHECK: [
  // CHECK: [
  // CHECK: [4{{(, 4)*}}]
  call @printMemrefF32(%printed_m5) : (memref<*xf32>) -> ()

  return
}
