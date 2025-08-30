// RUN: buddy-opt %s --allow-unregistered-dialect -o - | FileCheck %s

// Test basic dynamic vector type parsing and printing
func.func @test_dynamic_vector_types() {
  // CHECK: %0 = "test.op"() : () -> !vir.vec<?xf32>
  %0 = "test.op"() : () -> !vir.vec<?xf32>
  // CHECK: %1 = "test.op"() : () -> !vir.vec<2x?xi32>
  %1 = "test.op"() : () -> !vir.vec<2x?xi32>
  // CHECK: %2 = "test.op"() : () -> !vir.vec<?xi8, #vir.sf<m4>>
  %2 = "test.op"() : () -> !vir.vec<?xi8, m4>
  // CHECK: %3 = "test.op"() : () -> !vir.vec<?xi64, #vir.sf<f2>>
  %3 = "test.op"() : () -> !vir.vec<?xi64, f2>
  // CHECK: %4 = "test.op"() : () -> !vir.vec<?xi64, #vir.sf<f2>>
  %4 = "test.op"() : () -> !vir.vec<?xi64, #vir.sf<f2>>

  return
}
