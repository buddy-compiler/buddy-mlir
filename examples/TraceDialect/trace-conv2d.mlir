// RUN: buddy-opt %s --lower-trace | FileCheck %s 

#map = affine_map<(d0, d1) -> (d0 + d1 - 1)>

module {
  func.func private @rtclock() -> f64
  func.func private @printF64(f64)
  func.func private @printNewline()
  
  func.func @alloc_2d_filled_f32(%rows: index, %cols: index, %value: f32) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc(%rows, %cols) : memref<?x?xf32>
    scf.for %i = %c0 to %rows step %c1 {
      scf.for %j = %c0 to %cols step %c1 {
        memref.store %value, %alloc[%i, %j] : memref<?x?xf32>
      }
    }
    return %alloc : memref<?x?xf32>
  }

  func.func @main() {
    %cst_one = arith.constant 1.0 : f32
    %cst_zero = arith.constant 0.0 : f32
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %c10 = arith.constant 10 : index 

    %filter = call @alloc_2d_filled_f32(%c3, %c3, %cst_one) : (index, index, f32) -> memref<?x?xf32>
    %input = call @alloc_2d_filled_f32(%c10, %c10, %cst_one) : (index, index, f32) -> memref<?x?xf32>
    %output = call @alloc_2d_filled_f32(%c8, %c8, %cst_zero) : (index, index, f32) -> memref<?x?xf32>
    // start timing 
    // CHECK: call @rtclock() : () -> f64
    trace.time_start
    linalg.conv_2d ins(%input, %filter : memref<?x?xf32>, memref<?x?xf32>) outs(%output : memref<?x?xf32>)
    // end timing
    // CHECK: call @rtclock() : () -> f64
    trace.time_end

    // %conv2d = trace.scope -> (memref<?x?xf32>) {
    //   linalg.conv_2d ins(%input, %filter : memref<?x?xf32>, memref<?x?xf32>) outs(%output : memref<?x?xf32>)
    //   trace.yield %output : memref<?x?xf32>
    // }
    return
  }
}