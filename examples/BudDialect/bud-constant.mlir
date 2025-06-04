// RUN: buddy-opt %s -lower-bud \
// RUN: | FileCheck %s

module {
  // CHECK: %{{.*}} = arith.constant 0 : i32
  %i0 = bud.test_constant : i32
}
