// RUN: buddy-opt %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:		-convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:		-convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>

module{
  memref.global "private" @kernel : memref<3x3xf32> = dense<0.0>

  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @pooling_nhwc_max(%a : memref<?x?x?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?x?x?xf32>) {
    linalg.pooling_nhwc_max  
      ins(%a, %b : memref<?x?x?x?xf32>, memref<?x?xf32>) 
      outs(%c : memref<?x?x?x?xf32>)
    return
  }

  func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    scf.for %arg5 = %c0 to %arg0 step %c1 {
      scf.for %arg6 = %c0 to %arg1 step %c1 {
        scf.for %arg7 = %c0 to %arg2 step %c1 {
          scf.for %arg8 = %c0 to %arg3 step %c1 {
            memref.store %arg4, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }

  func.func @main(){
    %N = arith.constant 1 : index
    %current_v1 = arith.constant 3 : index
    %current_v2 = arith.constant 126 : index
    %current_v0 = affine.apply #map0(%current_v2, %current_v1)
    %c0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1.000000e+00 : f32
    %kernel = memref.get_global @kernel : memref<3x3xf32>

    %a = call @alloc_f32(%N, %current_v0, %current_v0, %N, %c1) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %b = memref.cast %kernel : memref<3x3xf32> to memref<?x?xf32>
    %c = call @alloc_f32(%N, %current_v2, %current_v2, %N, %c0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    %t0 = call @rtclock() : () -> f64
    call @pooling_nhwc_max(%a, %b, %c) : (memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>) -> ()
    %t1 = call @rtclock() : () -> f64
    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref
    // CHECK: [
    // CHECK: [
    // CHECK: [
    // CHECK: [1],
    %print_c = memref.cast %c : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_c) : (memref<*xf32>) -> ()

    %time = arith.subf %t1, %t0 : f64
    vector.print %time : f64

    memref.dealloc %a : memref<?x?x?x?xf32>
    memref.dealloc %c : memref<?x?x?x?xf32>

    return 
  }
}
