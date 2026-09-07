// RUN: buddy-opt %s --verify-diagnostics --split-input-file

module {
  func.func @bad_output_width(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq: memref<1x1x64x1024xi8>, %ws: memref<1x63xf32>,
      %output: memref<1x63xf32>) {
    // expected-error@+1 {{D must be divisible by OUTBLK (64)}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<1x1x64x1024xi8>, memref<1x63xf32>,
         memref<1x63xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_fused_scale_shape(
      %gate: memref<1x3072xf32>, %up: memref<1x3072xf32>,
      %q: memref<1x3072xi8>, %scales: memref<1x5xf32>) {
    // expected-error@+1 {{scales shape must be [T, K / group_size]}}
    "bosc_ame.silu_mul_quantize_per_group"(
        %gate, %up, %q, %scales) {group_size = 512 : i64} :
        (memref<1x3072xf32>, memref<1x3072xf32>,
         memref<1x3072xi8>, memref<1x5xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_fused_mixed_form(
      %gate: memref<1x3072xf32>, %up: memref<1x3072xf32>)
      -> (tensor<1x3072xi8>, tensor<1x6xf32>) {
    %q = tensor.empty() : tensor<1x3072xi8>
    %scales = tensor.empty() : tensor<1x6xf32>
    // expected-error@+1 {{silu/up and destinations must use the same tensor/buffer form}}
    %r:2 = "bosc_ame.silu_mul_quantize_per_group"(
        %gate, %up, %q, %scales) {group_size = 512 : i64} :
        (memref<1x3072xf32>, memref<1x3072xf32>,
         tensor<1x3072xi8>, tensor<1x6xf32>) ->
        (tensor<1x3072xi8>, tensor<1x6xf32>)
    return %r#0, %r#1 : tensor<1x3072xi8>, tensor<1x6xf32>
  }
}

// -----

module {
  func.func @bad_fused_vector_inputs(
      %gate: vector<1x3072xf32>, %up: vector<1x3072xf32>,
      %q: memref<1x3072xi8>, %scales: memref<1x6xf32>) {
    // expected-error@+1 {{silu/up and destinations must use the same tensor/buffer form}}
    "bosc_ame.silu_mul_quantize_per_group"(
        %gate, %up, %q, %scales) {group_size = 512 : i64} :
        (vector<1x3072xf32>, vector<1x3072xf32>,
         memref<1x3072xi8>, memref<1x6xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_layout(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq: memref<1x1x64x1024xi8>, %ws: memref<1x64xf32>,
      %output: memref<1x64xf32>) {
    // expected-error@+1 {{weight_layout must be 'ame_outblk64'}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 1024 : i64, weight_layout = "row_major"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<1x1x64x1024xi8>, memref<1x64xf32>,
         memref<1x64xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_group_divisibility(
      %xq: memref<1x1024xi8>, %xs: memref<1x4xf32>,
      %wq: memref<1x4x64x300xi8>, %ws: memref<4x64xf32>,
      %output: memref<1x64xf32>) {
    // expected-error@+1 {{K must be divisible by group_size}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 300 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x4xf32>,
         memref<1x4x64x300xi8>, memref<4x64xf32>,
         memref<1x64xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_k64(
      %xq: memref<1x1000xi8>, %xs: memref<1x4xf32>,
      %wq: memref<1x4x64x250xi8>, %ws: memref<4x64xf32>,
      %output: memref<1x64xf32>) {
    // expected-error@+1 {{K must be divisible by the AME K tile (64)}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 250 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1000xi8>, memref<1x4xf32>,
         memref<1x4x64x250xi8>, memref<4x64xf32>,
         memref<1x64xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @group_too_large(
      %xq: memref<1x2048xi8>, %xs: memref<1x1xf32>,
      %wq: memref<1x1x64x2048xi8>, %ws: memref<1x64xf32>,
      %output: memref<1x64xf32>) {
    // expected-error@+1 {{group_size must be in [1, 1024]}}
    "bosc_ame.w8a8_linear"(%xq, %xs, %wq, %ws, %output)
        {group_size = 2048 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x2048xi8>, memref<1x1xf32>,
         memref<1x1x64x2048xi8>, memref<1x64xf32>,
         memref<1x64xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_pair_output_shape(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq0: memref<48x1x64x1024xi8>, %ws0: memref<1x3072xf32>,
      %wq1: memref<48x1x64x1024xi8>, %ws1: memref<1x3072xf32>,
      %out0: memref<1x3072xf32>, %out1: memref<1x2048xf32>) {
    // expected-error@+1 {{paired output shapes must match}}
    "bosc_ame.w8a8_linear_pair"(
        %xq, %xs, %wq0, %ws0, %wq1, %ws1, %out0, %out1)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<1x3072xf32>, memref<1x2048xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_pair_second_weight_shape(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq0: memref<48x1x64x1024xi8>, %ws0: memref<1x3072xf32>,
      %wq1: memref<47x1x64x1024xi8>, %ws1: memref<1x3072xf32>,
      %out0: memref<1x3072xf32>, %out1: memref<1x3072xf32>) {
    // expected-error@+1 {{both wq shapes must be [D / 64, K / group_size, 64, group_size]}}
    "bosc_ame.w8a8_linear_pair"(
        %xq, %xs, %wq0, %ws0, %wq1, %ws1, %out0, %out1)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<47x1x64x1024xi8>, memref<1x3072xf32>,
         memref<1x3072xf32>, memref<1x3072xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_pair_layout(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq0: memref<48x1x64x1024xi8>, %ws0: memref<1x3072xf32>,
      %wq1: memref<48x1x64x1024xi8>, %ws1: memref<1x3072xf32>,
      %out0: memref<1x3072xf32>, %out1: memref<1x3072xf32>) {
    // expected-error@+1 {{weight_layout must be 'ame_outblk64'}}
    "bosc_ame.w8a8_linear_pair"(
        %xq, %xs, %wq0, %ws0, %wq1, %ws1, %out0, %out1)
        {group_size = 1024 : i64, weight_layout = "row_major"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<1x3072xf32>, memref<1x3072xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_pair_same_output(
      %xq: memref<1x1024xi8>, %xs: memref<1x1xf32>,
      %wq0: memref<48x1x64x1024xi8>, %ws0: memref<1x3072xf32>,
      %wq1: memref<48x1x64x1024xi8>, %ws1: memref<1x3072xf32>,
      %out: memref<1x3072xf32>) {
    // expected-error@+1 {{buffer-form paired outputs must not alias}}
    "bosc_ame.w8a8_linear_pair"(
        %xq, %xs, %wq0, %ws0, %wq1, %ws1, %out, %out)
        {group_size = 1024 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x1024xi8>, memref<1x1xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<48x1x64x1024xi8>, memref<1x3072xf32>,
         memref<1x3072xf32>, memref<1x3072xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @bad_pair_group_tile(
      %xq: memref<1x192xi8>, %xs: memref<1x2xf32>,
      %wq0: memref<48x2x64x96xi8>, %ws0: memref<2x3072xf32>,
      %wq1: memref<48x2x64x96xi8>, %ws1: memref<2x3072xf32>,
      %out0: memref<1x3072xf32>, %out1: memref<1x3072xf32>) {
    // expected-error@+1 {{group_size must be divisible by the AME K tile (64)}}
    "bosc_ame.w8a8_linear_pair"(
        %xq, %xs, %wq0, %ws0, %wq1, %ws1, %out0, %out1)
        {group_size = 96 : i64, weight_layout = "ame_outblk64"} :
        (memref<1x192xi8>, memref<1x2xf32>,
         memref<48x2x64x96xi8>, memref<2x3072xf32>,
         memref<48x2x64x96xi8>, memref<2x3072xf32>,
         memref<1x3072xf32>, memref<1x3072xf32>) -> ()
    return
  }
}
