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
func.func private @printMemrefF32(tensor<*xf32>)

func.func @test(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>) {
  %t_start = call @rtclock() : () -> f64
  %0 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %1 = tosa.transpose %arg0, %0 : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %2 = tosa.mul %1, %arg1 {shift = 0 : i8} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  // Print timings.
  vector.print %time : f64
  return %2 : tensor<?x?x?x?xf32>
}

// func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
//   scf.for %idx0 = %c0 to %arg0 step %c1 {
//     scf.for %idx1 = %c0 to %arg1 step %c1 {
//       scf.for %idx2 = %c0 to %arg2 step %c1 {
//         scf.for %idx3 = %c0 to %arg3 step %c1 {
//           memref.store %arg4, %0[%idx0, %idx1, %idx2, %idx3] : memref<?x?x?x?xf32>
//         }
//       }
//     }
//   }
//   return %0 : memref<?x?x?x?xf32>
// }

func.func @main(){
  // %c32 = arith.constant 32 : index
  // %c40 = arith.constant 40 : index
  // %c64 = arith.constant 64 : index
  // %c1 = arith.constant 1 : index
  // %c3 = arith.constant 3 : index
  // %c4 = arith.constant 4 : index
  // %f0 = arith.constant 0.0 : f32
  // %f2 = arith.constant 2.0 : f32

  // %m0 = call @alloc_f32(%c1, %c40, %c32, %c64, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  // %m1 = call @alloc_f32(%c1, %c1, %c40, %c64, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  // %m2 = call @alloc_f32(%c1, %c32, %c40, %c64, %f0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

  %v0 = arith.constant dense<2.0> : tensor<1x40x32x128xf32>
  %v1 = arith.constant dense<2.0> : tensor<1x1x40x128xf32>
  %tmp_v0 = tensor.cast %v0 : tensor<1x40x32x128xf32> to tensor<?x?x?x?xf32>
  %tmp_v1 = tensor.cast %v1 : tensor<1x1x40x128xf32> to tensor<?x?x?x?xf32>

  %v2 = call @test(%tmp_v0, %tmp_v1) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)

  %printed_v2 = tensor.cast %v2 : tensor<?x?x?x?xf32> to tensor<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 32, 40, 128] strides = [163840, 5120, 128, 1] data = 
  // CHECK-NEXT: [
  // CHECK: [
  // CHECK: [
  // CHECK: [4{{(, 4)*}}]
  call @printMemrefF32(%printed_v2) : (tensor<*xf32>) -> ()

  // %m3 = call @alloc_f32(%c1, %c3, %c3, %c32, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  // %m4 = call @alloc_f32(%c1, %c1, %c3, %c32, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  // %m5 = call @alloc_f32(%c1, %c3, %c3, %c32, %f0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

  %v3 = arith.constant dense<2.0> : tensor<1x4x3x32xf32>
  %v4 = arith.constant dense<2.0> : tensor<1x1x4x32xf32>
  %tmp_v3 = tensor.cast %v3 : tensor<1x4x3x32xf32> to tensor<?x?x?x?xf32>
  %tmp_v4 = tensor.cast %v4 : tensor<1x1x4x32xf32> to tensor<?x?x?x?xf32>

  %v5 = call @test(%tmp_v3, %tmp_v4) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)

  %printed_v5 = tensor.cast %v5 : tensor<?x?x?x?xf32> to tensor<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 3, 4, 32] strides = [384, 128, 32, 1] data =
  // CHECK-NEXT: [
  // CHECK: [
  // CHECK: [
  // CHECK: [4{{(, 4)*}}]
  call @printMemrefF32(%printed_v5) : (tensor<*xf32>) -> ()

  return
}
