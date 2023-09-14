 func.func @main() {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %A = memref.alloc() : memref<16x32x64xf32>
  %B = memref.alloc() : memref<16x64xf32>

  linalg.reduce
      ins(%A:memref<16x32x64xf32>)
      outs(%B:memref<16x64xf32>)
      dimensions = [1]
      {sche.dimensions = [0], sche.devices = [{targetId = "cpu", targetConfig = "", duty_ratio = 0.2:f32}, {targetId = "gpu", targetConfig = "", duty_ratio = 0.8:f32}]}
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }

  %res = memref.load %B[%c1,%c2] : memref<16x64xf32>
  // CHECK: 10
  vector.print %res : f32

  memref.dealloc %A : memref<16x32x64xf32>
  memref.dealloc %B : memref<16x64xf32>

  return
}
 