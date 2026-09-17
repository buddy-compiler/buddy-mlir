// RUN: buddy-opt %s --cse | FileCheck %s

module {
  func.func @tensor_quantize_is_shared(
      %input: tensor<22x1024xf32>) ->
      (tensor<22x1024xi8>, tensor<22x1xf32>,
       tensor<22x1024xi8>, tensor<22x1xf32>) {
    %xq0_init = tensor.empty() : tensor<22x1024xi8>
    %xs0_init = tensor.empty() : tensor<22x1xf32>
    %xq0, %xs0 = "bosc_ame.quantize_per_group"(
        %input, %xq0_init, %xs0_init) {group_size = 1024 : i64} :
        (tensor<22x1024xf32>, tensor<22x1024xi8>, tensor<22x1xf32>) ->
        (tensor<22x1024xi8>, tensor<22x1xf32>)
    %xq1_init = tensor.empty() : tensor<22x1024xi8>
    %xs1_init = tensor.empty() : tensor<22x1xf32>
    %xq1, %xs1 = "bosc_ame.quantize_per_group"(
        %input, %xq1_init, %xs1_init) {group_size = 1024 : i64} :
        (tensor<22x1024xf32>, tensor<22x1024xi8>, tensor<22x1xf32>) ->
        (tensor<22x1024xi8>, tensor<22x1xf32>)
    return %xq0, %xs0, %xq1, %xs1 :
        tensor<22x1024xi8>, tensor<22x1xf32>,
        tensor<22x1024xi8>, tensor<22x1xf32>
  }

  func.func @buffer_quantize_writes_are_preserved(
      %input: memref<22x1024xf32>,
      %xq: memref<22x1024xi8>, %xs: memref<22x1xf32>) {
    "bosc_ame.quantize_per_group"(%input, %xq, %xs)
        {group_size = 1024 : i64} :
        (memref<22x1024xf32>, memref<22x1024xi8>, memref<22x1xf32>) -> ()
    "bosc_ame.quantize_per_group"(%input, %xq, %xs)
        {group_size = 1024 : i64} :
        (memref<22x1024xf32>, memref<22x1024xi8>, memref<22x1xf32>) -> ()
    return
  }

  func.func @buffer_fused_quantize_writes_are_preserved(
      %gate: memref<1x3072xf32>, %up: memref<1x3072xf32>,
      %xq: memref<1x3072xi8>, %xs: memref<1x6xf32>) {
    "bosc_ame.silu_mul_quantize_per_group"(%gate, %up, %xq, %xs)
        {group_size = 512 : i64} :
        (memref<1x3072xf32>, memref<1x3072xf32>,
         memref<1x3072xi8>, memref<1x6xf32>) -> ()
    "bosc_ame.silu_mul_quantize_per_group"(%gate, %up, %xq, %xs)
        {group_size = 512 : i64} :
        (memref<1x3072xf32>, memref<1x3072xf32>,
         memref<1x3072xi8>, memref<1x6xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @tensor_quantize_is_shared
// CHECK-COUNT-1: bosc_ame.quantize_per_group
// CHECK: return %[[XQ:.*]], %[[XS:.*]], %[[XQ]], %[[XS]]

// CHECK-LABEL: func.func @buffer_quantize_writes_are_preserved
// CHECK-COUNT-2: bosc_ame.quantize_per_group
// CHECK: return

// CHECK-LABEL: func.func @buffer_fused_quantize_writes_are_preserved
// CHECK-COUNT-2: bosc_ame.silu_mul_quantize_per_group
// CHECK: return
