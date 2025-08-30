// RUN: buddy-opt %s \
// RUN:     -batchmatmul-transpose-b-vectorization \
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
    linalg.batch_matmul_transpose_b
      ins(%a, %b: memref<?x?x?xf32>, memref<?x?x?xf32>)
      outs(%c: memref<?x?x?xf32>)
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
