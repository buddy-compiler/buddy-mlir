// next-silu-silu.mlir
//
// Manually optimized version of SiLU.
// This implementation avoids TOSA and uses lower-level dialects like
// scf, arith, math, and vector to have fine-grained control over the computation.
// The core idea is to vectorize the innermost loop.
//
#map = affine_map<(d0) -> (d0)>
module {
  // Declare external utility functions for timing and printing.
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(%ptr: memref<*xf32>) attributes {llvm.emit_c_interface}

  // The kernel function that performs the SiLU calculation.
  func.func @kernel_silu(%arg0: memref<1x40x8960xf32>) {
    %t_start = call @rtclock() : () -> f64
    
    // Allocate output buffer
    %output = memref.alloc() : memref<1x40x8960xf32>
    
    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst_1f = arith.constant 1.0 : f32
    %vec_1f = vector.broadcast %cst_1f : f32 to vector<8xf32>
    %cst_0f = arith.constant 0.0 : f32 // for padding

    // Get dimensions.
    %d0 = memref.dim %arg0, %c0 : memref<1x40x8960xf32>
    %d1 = memref.dim %arg0, %c1 : memref<1x40x8960xf32>
    %d2 = memref.dim %arg0, %c2 : memref<1x40x8960xf32>

    // Loop nest
    affine.for %i = #map(%c0) to #map(%d0) {
      affine.for %j = #map(%c0) to #map(%d1) {
        affine.for %k = #map(%c0) to #map(%d2) step 8 {
          // Vectorized SiLU computation
          %x_vec = vector.transfer_read %arg0[%i, %j, %k], %cst_0f : memref<1x40x8960xf32>, vector<8xf32>
          %neg_x_vec = arith.negf %x_vec : vector<8xf32>
          %exp_neg_x_vec = math.exp %neg_x_vec : vector<8xf32>
          %one_plus_exp_vec = arith.addf %vec_1f, %exp_neg_x_vec : vector<8xf32>
          %sigmoid_x_vec = arith.divf %vec_1f, %one_plus_exp_vec : vector<8xf32>
          %silu_vec = arith.mulf %x_vec, %sigmoid_x_vec : vector<8xf32>
          vector.transfer_write %silu_vec, %output[%i, %j, %k] : vector<8xf32>, memref<1x40x8960xf32>
        }
      }
    }

    %t_end = call @rtclock() : () -> f64
    %unranked_result = memref.cast %output : memref<1x40x8960xf32> to memref<*xf32>
    call @printMemrefF32(%unranked_result) : (memref<*xf32>) -> ()
    memref.dealloc %output : memref<1x40x8960xf32>
    
    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64
    
    return 
  }

  // Main function to drive the test.
  func.func @main() {
    // 1. Allocate and initialize input.
    %input = memref.alloc() : memref<1x40x8960xf32>
    %cst_neg_1_23 = arith.constant 3.0 : f32
    linalg.fill ins(%cst_neg_1_23 : f32) outs(%input : memref<1x40x8960xf32>)

    // 2. Call the SiLU kernel.
    call @kernel_silu(%input) : (memref<1x40x8960xf32>) -> ()


    // 4. Deallocate memrefs.
    memref.dealloc %input : memref<1x40x8960xf32>
    
    return
  }
}

