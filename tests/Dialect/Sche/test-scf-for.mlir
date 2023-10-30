// RUN: buddy-opt %s -device-schedule | FileCheck %s -check-prefix=CHECK_schedule
// RUN: buddy-opt %s -device-schedule -lower-sche | FileCheck %s -check-prefix=CHECK_lower

func.func @test_sche() {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  %lb = arith.constant 0 : index
  %ub = arith.constant 100 : index

  %A = memref.alloc() : memref<100xf32>
  %A_cast0 = memref.cast %A : memref<100xf32> to memref<*xf32>

  scf.for %iv = %lb to %ub step %c1 {
    memref.store %c0, %A[%iv] : memref<100xf32>
  }
  // CHECK_schedule: sche.on_device async {{.*}}sche.source = "scf.for", targetConfig = "", targetId = "gpu"
  // CHECK_schedule: sche.wait
  // CHECK_lower: async.execute {
  // CHECK_lower: gpu.launch
  // CHECK_lower: }
  // CHECK_lower: async.await
  scf.for %iv = %lb to %ub step %c1 {
    %0 = arith.muli %iv, %c8 : index
    %1 = arith.addi %iv, %0  : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = memref.load %A[%iv] : memref<100xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %A[%iv] : memref<100xf32>
  } {sche.devices = [{targetId = "cpu", targetConfig = "", duty_ratio = 0.2:f32}, {targetId = "gpu", targetConfig = "", duty_ratio = 0.8:f32}]}

  %res = memref.load %A[%c1] : memref<100xf32>
  call @printMemrefF32(%A_cast0) : (memref<*xf32>) -> ()

  memref.dealloc %A : memref<100xf32>

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
