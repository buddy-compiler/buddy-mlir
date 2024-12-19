// RUN: buddy-opt %s \
// RUN:   -pooling-nhwc-max-vectorization \
// RUN:   -convert-linalg-to-loops \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module{
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @pooling_nhwc_max(%a : memref<?x?x?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?x?x?xf32>) {
    linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} 
      ins(%a, %b : memref<?x?x?x?xf32>, memref<?x?xf32>) 
      outs(%c : memref<?x?x?x?xf32>)
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

  func.func @alloc2_f32(%arg0: index, %arg1: index, %arg4: f32) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %idx0 = %c0 to %arg0 step %c1 {
      scf.for %idx1 = %c0 to %arg1 step %c1 {   
        memref.store %arg4, %0[%idx0, %idx1] : memref<?x?xf32>
      }
    }
    return %0 : memref<?x?xf32>
  }

  func.func @main(){
    // Set up dims.
    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c2 = arith.constant 2 : index
    %c12 = arith.constant 12 : index
    %c6 = arith.constant 6 : index

    // Set Init Value.
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    %v0 = call @alloc_f32(%c1, %c24, %c24, %c6, %f1) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %v1 = call @alloc2_f32(%c2, %c2, %f0) : (index, index, f32) -> memref<?x?xf32>
    %v2 = call @alloc_f32(%c1, %c12, %c12, %c6, %f0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    %t0 = call @rtclock() : () -> f64
    call @pooling_nhwc_max(%v0, %v1, %v2) : (memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>) -> ()
    %t1 = call @rtclock() : () -> f64
    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref
    // CHECK: [
    // CHECK: [
    // CHECK: [
    // CHECK: [1{{(, 1)*}}],
    %print_C = memref.cast %v2 : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()
    %time = arith.subf %t1, %t0 : f64
    vector.print %time : f64

    memref.dealloc %v0 : memref<?x?x?x?xf32>
    memref.dealloc %v1 : memref<?x?xf32>
    memref.dealloc %v2 : memref<?x?x?x?xf32>

    return 
  }
}
