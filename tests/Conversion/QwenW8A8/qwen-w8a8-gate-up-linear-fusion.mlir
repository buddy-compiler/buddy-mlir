// RUN: buddy-opt %s --fuse-qwen-gate-up-w8a8-linear | FileCheck %s --check-prefix=FUSE
// RUN: buddy-opt %s --fuse-qwen-gate-up-w8a8-linear --tosa-to-tensor --eliminate-empty-tensors --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries" | FileCheck %s --check-prefix=BUFFER

// This is the exact Qwen3 decode Gate/Up geometry.  The two projections
// intentionally use the same tensor.empty destination, as the frontend does
// after empty tensor elimination.  The paired destination-style op must
// nevertheless bufferize its two results into distinct writable allocations.
func.func @qwen_gate_up_t1(
    %xq: tensor<1x1024xi8>, %xs: tensor<1x1xf32>,
    %gate_wq: tensor<48x1x64x1024xi8>,
    %gate_ws: tensor<1x3072xf32>,
    %up_wq: tensor<48x1x64x1024xi8>,
    %up_ws: tensor<1x3072xf32>)
    -> (tensor<1x3072xf32>, tensor<1x3072xf32>,
        tensor<1x1x3072xf32>) {
  %init = tensor.empty() : tensor<1x3072xf32>
  %gate = "bosc_ame.w8a8_linear"(
      %xq, %xs, %gate_wq, %gate_ws, %init)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<1x1024xi8>, tensor<1x1xf32>,
       tensor<48x1x64x1024xi8>, tensor<1x3072xf32>,
       tensor<1x3072xf32>) -> tensor<1x3072xf32>
  // The frontend defines the pure reshape shape and its Gate-result consumer
  // between Gate and Up.
  %shape = tosa.const_shape {values = dense<[1, 1, 3072]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %gate3 = tosa.reshape %gate, %shape :
      (tensor<1x3072xf32>, !tosa.shape<3>) -> tensor<1x1x3072xf32>
  %up = "bosc_ame.w8a8_linear"(
      %xq, %xs, %up_wq, %up_ws, %init)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<1x1024xi8>, tensor<1x1xf32>,
       tensor<48x1x64x1024xi8>, tensor<1x3072xf32>,
       tensor<1x3072xf32>) -> tensor<1x3072xf32>
  return %gate, %up, %gate3 : tensor<1x3072xf32>,
      tensor<1x3072xf32>, tensor<1x1x3072xf32>
}

// FUSE-LABEL: func.func @qwen_gate_up_t1
// FUSE: %[[INIT:.*]] = tensor.empty() : tensor<1x3072xf32>
// FUSE: %[[PAIR:.*]]:2 = "bosc_ame.w8a8_linear_pair"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %[[INIT]], %[[INIT]])
// FUSE-SAME: group_size = 1024
// FUSE-SAME: weight_layout = "ame_outblk64"
// FUSE: tosa.reshape %[[PAIR]]#0
// FUSE-NOT: "bosc_ame.w8a8_linear"
// FUSE: return %[[PAIR]]#0, %[[PAIR]]#1

// BUFFER-LABEL: func.func @qwen_gate_up_t1
// BUFFER-DAG: %[[OUT0:.*]] = memref.alloc() {{.*}} : memref<1x3072xf32>
// BUFFER-DAG: %[[OUT1:.*]] = memref.alloc() {{.*}} : memref<1x3072xf32>
// BUFFER: "bosc_ame.w8a8_linear_pair"({{.*}}, %[[OUT0]], %[[OUT1]])
// BUFFER-SAME: group_size = 1024
// BUFFER-NOT: "bosc_ame.w8a8_linear"(
// BUFFER: return

// The same Gate/Up geometry at T=22 intentionally remains two independent
// projections.  Sharing both output clears was checkpoint-correct but caused
// a repeatable FPGA prefill regression by disturbing output-cache locality.
func.func @qwen_gate_up_prefill_t22(
    %xq: tensor<22x1024xi8>, %xs: tensor<22x1xf32>,
    %wq0: tensor<48x1x64x1024xi8>, %ws0: tensor<1x3072xf32>,
    %wq1: tensor<48x1x64x1024xi8>, %ws1: tensor<1x3072xf32>)
    -> (tensor<22x3072xf32>, tensor<22x3072xf32>) {
  %init0 = tensor.empty() : tensor<22x3072xf32>
  %init1 = tensor.empty() : tensor<22x3072xf32>
  %a = "bosc_ame.w8a8_linear"(%xq, %xs, %wq0, %ws0, %init0)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<22x1024xi8>, tensor<22x1xf32>,
       tensor<48x1x64x1024xi8>, tensor<1x3072xf32>,
       tensor<22x3072xf32>) -> tensor<22x3072xf32>
  %b = "bosc_ame.w8a8_linear"(%xq, %xs, %wq1, %ws1, %init1)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<22x1024xi8>, tensor<22x1xf32>,
       tensor<48x1x64x1024xi8>, tensor<1x3072xf32>,
       tensor<22x3072xf32>) -> tensor<22x3072xf32>
  return %a, %b : tensor<22x3072xf32>, tensor<22x3072xf32>
}

// FUSE-LABEL: func.func @qwen_gate_up_prefill_t22
// FUSE-COUNT-2: "bosc_ame.w8a8_linear"(
// FUSE-NOT: bosc_ame.w8a8_linear_pair

// D=2048 is Q projection geometry, not Gate/Up, so two otherwise compatible
// linears must not be paired.
func.func @wrong_output_geometry(
    %xq: tensor<1x1024xi8>, %xs: tensor<1x1xf32>,
    %wq0: tensor<32x1x64x1024xi8>, %ws0: tensor<1x2048xf32>,
    %wq1: tensor<32x1x64x1024xi8>, %ws1: tensor<1x2048xf32>)
    -> (tensor<1x2048xf32>, tensor<1x2048xf32>) {
  %init0 = tensor.empty() : tensor<1x2048xf32>
  %init1 = tensor.empty() : tensor<1x2048xf32>
  %a = "bosc_ame.w8a8_linear"(%xq, %xs, %wq0, %ws0, %init0)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<1x1024xi8>, tensor<1x1xf32>, tensor<32x1x64x1024xi8>,
       tensor<1x2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
  %b = "bosc_ame.w8a8_linear"(%xq, %xs, %wq1, %ws1, %init1)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<1x1024xi8>, tensor<1x1xf32>, tensor<32x1x64x1024xi8>,
       tensor<1x2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
  return %a, %b : tensor<1x2048xf32>, tensor<1x2048xf32>
}

// FUSE-LABEL: func.func @wrong_output_geometry
// FUSE-COUNT-2: "bosc_ame.w8a8_linear"(
// FUSE-NOT: bosc_ame.w8a8_linear_pair

// Exact Gate/Up shapes are insufficient: both projections must consume the
// same quantized activation and activation-scale SSA values.
func.func @different_activation(
    %xq0: tensor<1x1024xi8>, %xs0: tensor<1x1xf32>,
    %xq1: tensor<1x1024xi8>, %xs1: tensor<1x1xf32>,
    %wq0: tensor<48x1x64x1024xi8>, %ws0: tensor<1x3072xf32>,
    %wq1: tensor<48x1x64x1024xi8>, %ws1: tensor<1x3072xf32>)
    -> (tensor<1x3072xf32>, tensor<1x3072xf32>) {
  %init0 = tensor.empty() : tensor<1x3072xf32>
  %init1 = tensor.empty() : tensor<1x3072xf32>
  %a = "bosc_ame.w8a8_linear"(%xq0, %xs0, %wq0, %ws0, %init0)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<1x1024xi8>, tensor<1x1xf32>, tensor<48x1x64x1024xi8>,
       tensor<1x3072xf32>, tensor<1x3072xf32>) -> tensor<1x3072xf32>
  %b = "bosc_ame.w8a8_linear"(%xq1, %xs1, %wq1, %ws1, %init1)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<1x1024xi8>, tensor<1x1xf32>, tensor<48x1x64x1024xi8>,
       tensor<1x3072xf32>, tensor<1x3072xf32>) -> tensor<1x3072xf32>
  return %a, %b : tensor<1x3072xf32>, tensor<1x3072xf32>
}

// FUSE-LABEL: func.func @different_activation
// FUSE-COUNT-2: "bosc_ame.w8a8_linear"(
// FUSE-NOT: bosc_ame.w8a8_linear_pair

// An unrelated reshape between the projections is not a legal scheduling
// boundary to cross: its result could define an operand of later work.
func.func @intervening_unrelated_reshape(
    %xq: tensor<1x1024xi8>, %xs: tensor<1x1xf32>,
    %wq0: tensor<48x1x64x1024xi8>, %ws0: tensor<1x3072xf32>,
    %wq1: tensor<48x1x64x1024xi8>, %ws1: tensor<1x3072xf32>,
    %side: tensor<1x3072xf32>)
    -> (tensor<1x3072xf32>, tensor<1x3072xf32>,
        tensor<1x1x3072xf32>) {
  %shape = tosa.const_shape {values = dense<[1, 1, 3072]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %init0 = tensor.empty() : tensor<1x3072xf32>
  %init1 = tensor.empty() : tensor<1x3072xf32>
  %a = "bosc_ame.w8a8_linear"(%xq, %xs, %wq0, %ws0, %init0)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<1x1024xi8>, tensor<1x1xf32>, tensor<48x1x64x1024xi8>,
       tensor<1x3072xf32>, tensor<1x3072xf32>) -> tensor<1x3072xf32>
  %side3 = tosa.reshape %side, %shape :
      (tensor<1x3072xf32>, !tosa.shape<3>) -> tensor<1x1x3072xf32>
  %b = "bosc_ame.w8a8_linear"(%xq, %xs, %wq1, %ws1, %init1)
      {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
      (tensor<1x1024xi8>, tensor<1x1xf32>, tensor<48x1x64x1024xi8>,
       tensor<1x3072xf32>, tensor<1x3072xf32>) -> tensor<1x3072xf32>
  return %a, %b, %side3 : tensor<1x3072xf32>,
      tensor<1x3072xf32>, tensor<1x1x3072xf32>
}

// FUSE-LABEL: func.func @intervening_unrelated_reshape
// FUSE-COUNT-2: "bosc_ame.w8a8_linear"(
// FUSE-NOT: bosc_ame.w8a8_linear_pair
