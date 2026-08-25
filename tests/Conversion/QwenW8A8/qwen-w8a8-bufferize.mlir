// RUN: buddy-opt %s --eliminate-empty-tensors --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries" --expand-strided-metadata --lower-qwen-w8a8-to-boscame | FileCheck %s

module {
  func.func @decode_tensor_form(
      %input: tensor<1x1024xf32>,
      %wq: tensor<1x1x64x1024xi8>,
      %ws: tensor<1x64xf32>) -> tensor<1x64xf32> {
    %xq_init = tensor.empty() : tensor<1x1024xi8>
    %xs_init = tensor.empty() : tensor<1x1xf32>
    %xq, %xs = "bosc_ame.quantize_per_group"(
        %input, %xq_init, %xs_init) {group_size = 1024 : i64} :
        (tensor<1x1024xf32>, tensor<1x1024xi8>, tensor<1x1xf32>) ->
        (tensor<1x1024xi8>, tensor<1x1xf32>)
    %output_init = tensor.empty() : tensor<1x64xf32>
    %output = "bosc_ame.w8a8_linear"(
        %xq, %xs, %wq, %ws, %output_init)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (tensor<1x1024xi8>, tensor<1x1xf32>,
         tensor<1x1x64x1024xi8>, tensor<1x64xf32>, tensor<1x64xf32>) ->
        tensor<1x64xf32>
    return %output : tensor<1x64xf32>
  }
}

// CHECK-LABEL: func.func @decode_tensor_form
// CHECK-NOT: bosc_ame.quantize_per_group
// CHECK-NOT: bosc_ame.w8a8_linear
// CHECK: memref.alloc
// CHECK: bosc_ame.msettilem
// CHECK: bosc_ame.mqma.b.mm
// CHECK: return
