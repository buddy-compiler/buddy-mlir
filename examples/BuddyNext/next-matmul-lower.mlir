module {
  memref.global "private" constant @__constant_3072x1536xf32 : memref<3072x1536xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_40x3072xf32 : memref<40x3072xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_40x1536xf32 : memref<40x1536xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @kernel(%arg0: tensor<40x3072xf32>, %arg1: tensor<3072x1536xf32>) {
    %c3072 = arith.constant 3072 : index
    %c1536 = arith.constant 1536 : index
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg1 : tensor<3072x1536xf32> to memref<3072x1536xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : tensor<40x3072xf32> to memref<40x3072xf32, strided<[?, ?], offset: ?>>
    %2 = call @rtclock() : () -> f64
    %3 = memref.get_global @__constant_40x1536xf32 : memref<40x1536xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<40x1536xf32>
    memref.copy %3, %alloc : memref<40x1536xf32> to memref<40x1536xf32>
    scf.for %arg2 = %c0 to %c40 step %c1 {
      scf.for %arg3 = %c0 to %c1536 step %c1 {
        scf.for %arg4 = %c0 to %c3072 step %c1 {
          %7 = memref.load %1[%arg2, %arg4] : memref<40x3072xf32, strided<[?, ?], offset: ?>>
          %8 = memref.load %0[%arg4, %arg3] : memref<3072x1536xf32, strided<[?, ?], offset: ?>>
          %9 = memref.load %alloc[%arg2, %arg3] : memref<40x1536xf32>
          %10 = arith.mulf %7, %8 : f32
          %11 = arith.addf %9, %10 : f32
          memref.store %11, %alloc[%arg2, %arg3] : memref<40x1536xf32>
        }
      }
    }
    %4 = call @rtclock() : () -> f64
    %5 = arith.subf %4, %2 : f64
    %cast = memref.cast %alloc : memref<40x1536xf32> to memref<*xf32>
    %6 = bufferization.to_tensor %cast : memref<*xf32> to tensor<*xf32>
    call @printMemrefF32(%6) : (tensor<*xf32>) -> ()
    vector.print %5 : f64
    return
  }
  func.func @main() {
    %0 = memref.get_global @__constant_40x3072xf32 : memref<40x3072xf32>
    %1 = bufferization.to_tensor %0 : memref<40x3072xf32> to tensor<40x3072xf32>
    %2 = memref.get_global @__constant_3072x1536xf32 : memref<3072x1536xf32>
    %3 = bufferization.to_tensor %2 : memref<3072x1536xf32> to tensor<3072x1536xf32>
    call @kernel(%1, %3) : (tensor<40x3072xf32>, tensor<3072x1536xf32>) -> ()
    return
  }
}

