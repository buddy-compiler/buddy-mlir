// RUN: buddy-opt %s \
// RUN:   -pooling-nhwc-max-vectorization="vector-size=64" \
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
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @printMemrefI32(memref<*xi32>)

  func.func @pooling_nhwc_max_f32(%a : memref<?x?x?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?x?x?xf32>) {
    linalg.pooling_nhwc_max  
      ins(%a, %b : memref<?x?x?x?xf32>, memref<?x?xf32>) 
      outs(%c : memref<?x?x?x?xf32>)
    return
  }

  func.func @pooling_nhwc_max_i32(%a : memref<?x?x?x?xi32>, %b : memref<?x?xi32>, %c : memref<?x?x?x?xi32>) {
    linalg.pooling_nhwc_max  
      ins(%a, %b : memref<?x?x?x?xi32>, memref<?x?xi32>) 
      outs(%c : memref<?x?x?x?xi32>)
    return
  }

  func.func @main(){
    // Set up dims.
    %c1 = arith.constant 1 : index
    %cInput = arith.constant 128 : index
    %cKernel = arith.constant 3 : index
    %cOutput = arith.constant 126 : index

    // -------------------------------------------------------------------------
    // Test f32 as element type.
    // -------------------------------------------------------------------------

    // Set Init Value.
    %cf1_32 = arith.constant 1.0 : f32

    %a_f32 = memref.alloc(%c1, %cInput, %cInput, %c1) : memref<?x?x?x?xf32>
    %b_f32 = memref.alloc(%cKernel, %cKernel) : memref<?x?xf32>
    %c_f32 = memref.alloc(%c1, %cOutput, %cOutput, %c1) : memref<?x?x?x?xf32>

    linalg.fill ins(%cf1_32 : f32) outs(%a_f32 : memref<?x?x?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%b_f32 : memref<?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%c_f32 : memref<?x?x?x?xf32>)

    call @pooling_nhwc_max_f32(%a_f32, %b_f32, %c_f32) : (memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>) -> ()
    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref
    // CHECK: [
    // CHECK: [
    // CHECK: [
    // CHECK: [1],
    %print_C_f32 = memref.cast %c_f32 : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C_f32) : (memref<*xf32>) -> ()

    memref.dealloc %c_f32 : memref<?x?x?x?xf32>
    memref.dealloc %b_f32 : memref<?x?xf32>
    memref.dealloc %a_f32 : memref<?x?x?x?xf32>

    // -------------------------------------------------------------------------
    // Test i32 as element type.
    // -------------------------------------------------------------------------

    // Set Init Value.
    %ci1_32 = arith.constant 1 : i32

    %a_i32 = memref.alloc(%c1, %cInput, %cInput, %c1) : memref<?x?x?x?xi32>
    %b_i32 = memref.alloc(%cKernel, %cKernel) : memref<?x?xi32>
    %c_i32 = memref.alloc(%c1, %cOutput, %cOutput, %c1) : memref<?x?x?x?xi32>

    linalg.fill ins(%ci1_32 : i32) outs(%a_i32 : memref<?x?x?x?xi32>)
    linalg.fill ins(%ci1_32 : i32) outs(%b_i32 : memref<?x?xi32>)
    linalg.fill ins(%ci1_32 : i32) outs(%c_i32 : memref<?x?x?x?xi32>)

    call @pooling_nhwc_max_i32(%a_i32, %b_i32, %c_i32) : (memref<?x?x?x?xi32>, memref<?x?xi32>, memref<?x?x?x?xi32>) -> ()
    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref
    // CHECK: [
    // CHECK: [
    // CHECK: [
    // CHECK: [1],
    %print_C_i32 = memref.cast %c_i32 : memref<?x?x?x?xi32> to memref<*xi32>
    call @printMemrefI32(%print_C_i32) : (memref<*xi32>) -> ()

    memref.dealloc %c_i32 : memref<?x?x?x?xi32>
    memref.dealloc %b_i32 : memref<?x?xi32>
    memref.dealloc %a_i32 : memref<?x?x?x?xi32>

    return 
  }
}
