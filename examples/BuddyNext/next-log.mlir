module {
  memref.global "private" constant @__constant_12x40x40xf32 : memref<12x40x40xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @kernel(%arg0: memref<12x40x40xf32>) {
    %cast = memref.cast %arg0 : memref<12x40x40xf32> to memref<12x40x40xf32, strided<[?, ?, ?], offset: ?>>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = call @rtclock() : () -> f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<12x40xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<12x40xf32>)
    linalg.reduce ins(%cast : memref<12x40x40xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc : memref<12x40xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %3 = arith.addf %in, %init : f32
        linalg.yield %3 : f32
      }
    // %expand_shape = memref.expand_shape %alloc [[0], [1, 2]] : memref<12x40xf32> into memref<12x40x1xf32>
    %1 = call @rtclock() : () -> f64
    %2 = arith.subf %1, %0 : f64
    // %cast_0 = memref.cast %expand_shape : memref<12x40x1xf32> to memref<*xf32>
    %cast_0 = memref.cast %alloc : memref<12x40xf32> to memref<*xf32>
    call @printMemrefF32(%cast_0) : (memref<*xf32>) -> ()
    vector.print %2 : f64
    return
  }
  func.func @main() {
    %0 = memref.get_global @__constant_12x40x40xf32 : memref<12x40x40xf32>
    call @kernel(%0) : (memref<12x40x40xf32>) -> ()
    return
  }
}

