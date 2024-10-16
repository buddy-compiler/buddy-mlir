// RUN: buddy-opt %s \
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
    linalg.pooling_nhwc_max  
      ins(%a, %b : memref<?x?x?x?xf32>, memref<?x?xf32>) 
      outs(%c : memref<?x?x?x?xf32>)
    return
  }

  func.func @main(){
    // Set up dims.
    %c1 = arith.constant 1 : index
    %cInput = arith.constant 128 : index
    %cKernel = arith.constant 3 : index
    %cOutput = arith.constant 126 : index

    // Set Init Value.
    %cf1_32 = arith.constant 1.0 : f32

    %a = memref.alloc(%c1, %cInput, %cInput, %c1) : memref<?x?x?x?xf32>
    %b = memref.alloc(%cKernel, %cKernel) : memref<?x?xf32>
    %c = memref.alloc(%c1, %cOutput, %cOutput, %c1) : memref<?x?x?x?xf32>

    linalg.fill ins(%cf1_32 : f32) outs(%a : memref<?x?x?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%b : memref<?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%c : memref<?x?x?x?xf32>)

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
    %print_C = memref.cast %c : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()
    %time = arith.subf %t1, %t0 : f64
    vector.print %time : f64

    memref.dealloc %c : memref<?x?x?x?xf32>
    memref.dealloc %b : memref<?x?xf32>
    memref.dealloc %a : memref<?x?x?x?xf32>

    return 
  }
}
