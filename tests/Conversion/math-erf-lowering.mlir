// RUN: buddy-opt %s -lower-linalg-to-vir -lower-vir-to-vector="vector-width=4" | FileCheck %s

// Verify math.erf lowering scalarization.
// CHECK: func.func private @erff(f32) -> f32

#map = affine_map<(d0) -> (d0)>

module {
  // Vector case
  // CHECK-LABEL: func.func @case_math_erf_vector
  // CHECK: %[[VEC:.*]] = vector.load
  // CHECK: %[[E0:.*]] = vector.extract %[[VEC]][0]
  // CHECK: %[[R0:.*]] = func.call @erff(%[[E0]]) : (f32) -> f32
  // CHECK: %[[I0:.*]] = vector.insert %[[R0]], %{{.*}} [0]
  // CHECK: %[[E1:.*]] = vector.extract %[[VEC]][1]
  // CHECK: %[[R1:.*]] = func.call @erff(%[[E1]]) : (f32) -> f32
  // CHECK: %[[I1:.*]] = vector.insert %[[R1]], %[[I0]] [1]
  // CHECK: %[[E2:.*]] = vector.extract %[[VEC]][2]
  // CHECK: %[[R2:.*]] = func.call @erff(%[[E2]]) : (f32) -> f32
  // CHECK: %[[I2:.*]] = vector.insert %[[R2]], %[[I1]] [2]
  // CHECK: %[[E3:.*]] = vector.extract %[[VEC]][3]
  // CHECK: %[[R3:.*]] = func.call @erff(%[[E3]]) : (f32) -> f32
  // CHECK: %[[I3:.*]] = vector.insert %[[R3]], %[[I2]] [3]
  // CHECK: vector.store %[[I3]]

  func.func @case_math_erf_vector(%arg0: memref<4xf32>, %arg1: memref<4xf32>) {
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
    ins(%arg0 : memref<4xf32>)
    outs(%arg1 : memref<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = math.erf %in : f32
      linalg.yield %0 : f32
    }
    return
  }

  // Scalar case
  // CHECK-LABEL: func.func @case_math_erf_scalar
  // CHECK: %[[S:.*]] = memref.load
  // CHECK: %[[SR:.*]] = func.call @erff(%[[S]]) : (f32) -> f32
  // CHECK: memref.store %[[SR]]

  func.func @case_math_erf_scalar(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
    ins(%arg0 : memref<1xf32>)
    outs(%arg1 : memref<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = math.erf %in : f32
      linalg.yield %0 : f32
    }
    return
  }

  // Ensure math.erf no longer survives after lowering.
  // CHECK-NOT: math.erf
}
