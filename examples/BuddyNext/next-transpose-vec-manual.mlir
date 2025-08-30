// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  memref.global "private" constant @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @kernel(%arg0: memref<1x32x40x128xf32>) {
    %0 = call @rtclock() : () -> f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x40x32x128xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 40 {
        affine.for %arg3 = 0 to 32 {
          affine.for %arg4 = 0 to 128 step 64 {
            %3 = vector.load %arg0[%arg1, %arg3, %arg2, %arg4] : memref<1x32x40x128xf32>, vector<64xf32>
            vector.store %3, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x40x32x128xf32>, vector<64xf32>
          }
        }
      }
    }
    %1 = call @rtclock() : () -> f64
    %2 = arith.subf %1, %0 : f64
    %cast = memref.cast %alloc : memref<1x40x32x128xf32> to memref<*xf32>

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 40, 32, 128] strides = [163840, 4096, 128, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [
    // CHECK-SAME: [3{{(, 3)*}}],

    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    vector.print %2 : f64
    return
  }
  func.func @main() {
    %0 = memref.get_global @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32>
    call @kernel(%0) : (memref<1x32x40x128xf32>) -> ()
    return
  }
}
