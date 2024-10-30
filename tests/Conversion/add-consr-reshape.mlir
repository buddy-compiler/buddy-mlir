
// RUN: buddy-opt -simplify-tosa-add-reshape %s | FileCheck %s

// CHECK: %0 = "tosa.const"() <{value = dense<7.000000e+00> : tensor<32x40x128xf32>}> : () -> tensor<32x40x128xf32>
// CHECK: return %0 : tensor<32x40x128xf32>
module {
    func.func @addconst() -> tensor<32x40x128xf32> {
        %0 = "tosa.const"() <{value = dense<3.5> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
        %1 = "tosa.const"() <{value = dense<3.5> : tensor<1x32x40x128xf32>}> : () -> tensor<1x32x40x128xf32>
        %2 = tosa.add %0, %1 : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
        %3 = tosa.reshape %2 {new_shape = array<i64: 32, 40, 128>} : (tensor<1x32x40x128xf32>) -> tensor<32x40x128xf32>
        return %3 : tensor<32x40x128xf32>
    }
}