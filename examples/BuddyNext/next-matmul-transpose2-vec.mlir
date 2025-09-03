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

func.func @test(%a : memref<?x?x?xf32>, %b : memref<?x?x?xf32>, %c : memref<?x?x?xf32>) {
  %t_start = call @rtclock() : () -> f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %vl_step = arith.constant 32 : index
  %c0_f32 = arith.constant 0.000000e+00 : f32
  %v0 = vector.splat %c0_f32 : vector<32xf32>
  %dim = memref.dim %a, %c0 : memref<?x?x?xf32>        //
  %dim_0 = memref.dim %a, %c1 : memref<?x?x?xf32>
  %dim_1 = memref.dim %a, %c2 : memref<?x?x?xf32>
  %dim_2 = memref.dim %b, %c2 : memref<?x?x?xf32>

  // Calculate the upper bound for vectorized processing
  // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
  // - Add 1 to ensure the final loop runs when the workload length
  //   is divisible by the vector size.
  %dim_2_upbound_tmp = arith.subi %dim_2, %vl_step : index
  %dim_2_upbound = arith.addi %dim_2_upbound_tmp, %c1 : index

  affine.for %arg3 = %c0 to %dim {
    affine.for %arg4 = %c0 to %dim_0 {
      %iter_idx = scf.for %arg5 = %c0 to %dim_2_upbound
          step %vl_step iter_args(%iter_init = %c0) -> (index){
        %0 = vector.load %c[%arg4, %arg3, %arg5] : memref<?x?x?xf32>, vector<32xf32>
        %iter_value = scf.for %arg6 = %c0 to %dim_1 step %c1 iter_args(%value_init = %0) -> (vector<32xf32>){
          %1 = memref.load %a[%arg3, %arg4, %arg6] : memref<?x?x?xf32>
          %2 = vector.splat %1 : vector<32xf32>
          %3 = vector.load %b[%arg6, %arg3, %arg5] : memref<?x?x?xf32>, vector<32xf32>
          %4 = vector.fma %2, %3, %value_init : vector<32xf32>
          scf.yield %4 : vector<32xf32>
        }
        vector.store %iter_value, %c[%arg4, %arg3, %arg5] : memref<?x?x?xf32>, vector<32xf32>
        %idx_next = arith.addi %arg5, %vl_step : index
        scf.yield %idx_next : index
      }

      // Compute the tail size and Process the remaining elements
      // using masked vector operations.
      %tail_size = arith.subi %dim_1, %iter_idx : index
      %mask = vector.create_mask %tail_size : vector<32xi1>
      %0 = vector.maskedload %c[%arg4, %arg3, %iter_idx], %mask, %v0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
      %iter_value = scf.for %arg6 = %c0 to %dim_1 step %c1 iter_args(%value_init = %0) -> (vector<32xf32>){
        %1 = memref.load %a[%arg3, %arg4, %arg6] : memref<?x?x?xf32>
        %2 = vector.splat %1 : vector<32xf32>
        %3 = vector.maskedload %b[%arg6, %arg3, %iter_idx], %mask, %v0 : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        %4 = vector.fma %2, %3, %value_init : vector<32xf32>
        scf.yield %4 : vector<32xf32>
      }
      vector.maskedstore %c[%arg4, %arg3, %iter_idx], %mask, %iter_value : memref<?x?x?xf32>, vector<32xi1>, vector<32xf32>
    }
  }

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  // Print timings.
  vector.print %time : f64
  return
}

func.func @alloc_f32(%dim0: index, %dim1: index, %dim2: index, %arg4: f32) -> memref<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%dim0, %dim1, %dim2) : memref<?x?x?xf32>
  scf.for %idx0 = %c0 to %dim0 step %c1 {
    scf.for %idx1 = %c0 to %dim1 step %c1 {
      scf.for %idx2 = %c0 to %dim2 step %c1 {
        memref.store %arg4, %0[%idx0, %idx1, %idx2] : memref<?x?x?xf32>
      }
    }
  }
  return %0 : memref<?x?x?xf32>
}


func.func @main(){
  %c32 = arith.constant 32 : index
  %c40 = arith.constant 40 : index
  %c128 = arith.constant 128 : index
  %f0 = arith.constant 0.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32

  %m0 = call @alloc_f32(%c32, %c40, %c40, %f2) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m1 = call @alloc_f32(%c40, %c32, %c128, %f3) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m2 = call @alloc_f32(%c40, %c32, %c128, %f0) : (index, index, index, f32) -> memref<?x?x?xf32>

  call @test(%m0, %m1, %m2) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()

  %printed_m2 = memref.cast %m2 : memref<?x?x?xf32> to memref<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [40, 32, 128] strides = [4096, 128, 1] data =
  // CHECK-NEXT: [
  // CHECK: [240{{(, 240)*}}]
  call @printMemrefF32(%printed_m2) : (memref<*xf32>) -> ()

  return
}
