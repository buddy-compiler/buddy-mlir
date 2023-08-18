module {
  func.func @forward() -> memref<1x13xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c13 = arith.constant 13 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x13xi1>
    %0 = bufferization.to_tensor %alloc : memref<1x13xi1>
    %1 = arith.extui %0 : tensor<1x13xi1> to tensor<1x13xi32>
    %2 = arith.bitcast %1 : tensor<1x13xi32> to tensor<1x13xf32>
    %3 = bufferization.to_memref %2 : memref<1x13xf32>
    %4 = bufferization.to_memref %2 : memref<1x13xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x13xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c13 step %c1 {
        %8 = memref.load %4[%arg0, %arg1] : memref<1x13xf32>
        %9 = memref.load %3[%arg0, %arg1] : memref<1x13xf32>
        %10 = arith.mulf %8, %9 : f32
        memref.store %10, %alloc_0[%arg0, %arg1] : memref<1x13xf32>
      }
    }
    %5 = bufferization.to_tensor %alloc_0 : memref<1x13xf32>
    %6 = math.rsqrt %5 : tensor<1x13xf32>
    %7 = bufferization.to_memref %6 : memref<1x13xf32>
    return %7 : memref<1x13xf32>
  }
}

