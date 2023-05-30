// RUN: buddy-opt %s -lower-bud \
// RUN: | FileCheck %s

module {
  // CHECK: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
  // CHECK: vector.print %[[CONSTANT_0]] : i32
  // CHECK: %[[VALUE_0:.*]] = vector.broadcast %[[CONSTANT_0]] : i32 to vector<4xi32>
  // CHECK: vector.print %[[VALUE_0]] : vector<4xi32>
  %i0 = bud.test_print : i32
}
