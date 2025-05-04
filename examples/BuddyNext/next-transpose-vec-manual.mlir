// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -func-bufferize \
// RUN:     -arith-bufferize \
// RUN:     -tensor-bufferize \
// RUN:     -buffer-deallocation \
// RUN:     -finalizing-bufferize \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  memref.global "private" constant @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @kernel(%arg0: memref<1x32x40x128xf32>) -> (){
    %alloc = memref.alloc() : memref<1x40x32x128xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %3 = vector.splat %cst : vector<64xf32>
    %dim = memref.dim %arg0, %c0 : memref<1x32x40x128xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<1x32x40x128xf32>
    %dim_1 = memref.dim %arg0, %c2 : memref<1x32x40x128xf32>
    %dim_2 = memref.dim %arg0, %c3 : memref<1x32x40x128xf32>

    %4 = arith.subi %dim_2, %c64 : index
    %5 = arith.addi %4, %c1 : index
    %t0 = call @rtclock() : () -> f64
    affine.for %arg1 = #map(%c0) to #map(%dim) {
      affine.for %arg2 = #map(%c0) to #map(%dim_1) {
        affine.for %arg3 = #map(%c0) to #map(%dim_0) {
          %8 = scf.for %arg4 = %c0 to %5 step %c64 iter_args(%arg5 = %c0) -> (index) {
            %12 = vector.load %arg0[%arg1, %arg3, %arg2, %arg4] : memref<1x32x40x128xf32>, vector<64xf32>
            vector.store %12, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x40x32x128xf32>, vector<64xf32>
            %13 = arith.addi %arg4, %c64 : index
            scf.yield %13 : index
          }
          %9 = arith.subi %dim_2, %8 : index
          %10 = vector.create_mask %9 : vector<64xi1>
          %11 = vector.maskedload %arg0[%arg1, %arg3, %arg2, %8], %10, %3 : memref<1x32x40x128xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          vector.maskedstore %alloc[%arg1, %arg2, %arg3, %8], %10, %11 : memref<1x40x32x128xf32>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    %t1 = call @rtclock() : () -> f64
    %20 = arith.subf %t1, %t0 : f64
    
    %cast = memref.cast %alloc : memref<1x40x32x128xf32> to memref<*xf32>

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 40, 32, 128] strides = [163840, 4096, 128, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [
    // CHECK-SAME: [3{{(, 3)*}}],

    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()

    vector.print %20 : f64
    return 
  }
  func.func @main() {
    %0 = memref.get_global @__constant_1x32x40x128xf32 : memref<1x32x40x128xf32>
    call @kernel(%0) : (memref<1x32x40x128xf32>) -> ()
    
    return
  }
}
