// RUN: buddy-opt %s -device-schedule | FileCheck %s -check-prefix=CHECK_schedule
// RUN: buddy-opt %s -device-schedule -lower-sche | FileCheck %s -check-prefix=CHECK_lower

func.func @main() -> vector<1x8x1x1xf32> attributes {sche.devices} {
  // CHECK_schedule: sche.on_device{{.*}}sche.source = "func", targetConfig = "", targetId = "gpu1"
  // CHECK_schedule: sche.on_device{{.*}}sche.source = "func", targetConfig = "", targetId = "gpu2"
  // CHECK_lower: gpu.launch {{.*}}{
  // CHECK_lower: vector.transfer_write
  // CHECK_lower: vector.transfer_write
  // CHECK_lower: }
  // CHECK_lower: vector.transfer_read
  // CHECK_lower: vector.transfer_read
  // CHECK_lower: vector.transfer_write
  // CHECK_lower: vector.transfer_write
  // CHECK_lower: gpu.launch {{.*}}{
  // CHECK_lower: vector.transfer_read
  // CHECK_lower: vector.transfer_read
  // CHECK_lower: vector.transfer_write
  // CHECK_lower: }
  // CHECK_lower: vector.transfer_read
  %0 = arith.constant dense<1.000000e-01> : vector<1x8x1x1xf32>
  %1 = arith.constant dense<0.000000e+00> : vector<1x8x1x1xf32>
  %2 = arith.addf %0, %1 : vector<1x8x1x1xf32>
  return %2 : vector<1x8x1x1xf32>
}
