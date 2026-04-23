func.func private @rtclock() -> f64

// Qwen3-0.6B-like prefill block (single layer, test-sized constants in main)
// hidden_size=1024, q_heads=16, kv_heads=8, head_dim=128, seq_len=512
func.func @qwen3_prefill_kernel(
    %x         : tensor<1x512x1024xf32>,
    %norm1_w   : tensor<1x1x1024xf32>,
    %wq        : tensor<1x1024x2048xf32>,
    %wk        : tensor<1x1024x1024xf32>,
    %wv        : tensor<1x1024x1024xf32>,
    %q_norm_w  : tensor<1x1x1x128xf32>,
    %k_norm_w  : tensor<1x1x1x128xf32>,
    %wo        : tensor<1x2048x1024xf32>,
    %attn_mask : tensor<1x1x512x512xf32>,
    %cos       : tensor<1x1x512x128xf32>,
    %sin       : tensor<1x1x512x128xf32>,
    %norm2_w   : tensor<1x1x1024xf32>,
    %ffn_gate  : tensor<1x1024x3072xf32>,
    %ffn_up    : tensor<1x1024x3072xf32>,
    %ffn_down  : tensor<1x3072x1024xf32>
) -> (tensor<1x512x1024xf32>, tensor<1x8x512x128xf32>, tensor<1x8x512x128xf32>) {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>

  // RMSNorm #1
  %x_sq = tosa.mul %x, %x, %shift : (tensor<1x512x1024xf32>, tensor<1x512x1024xf32>, tensor<1xi8>) -> tensor<1x512x1024xf32>
  %sum_sq = tosa.reduce_sum %x_sq {axis = 2 : i32} : (tensor<1x512x1024xf32>) -> tensor<1x512x1xf32>
  %n_inv = "tosa.const"() <{values = dense<9.765625e-04> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %mean_sq = tosa.mul %sum_sq, %n_inv, %shift : (tensor<1x512x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x512x1xf32>
  %eps = "tosa.const"() <{values = dense<1.0e-06> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %mean_sq_eps = tosa.add %mean_sq, %eps : (tensor<1x512x1xf32>, tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
  %rms_inv = tosa.rsqrt %mean_sq_eps : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
  %x_norm1 = tosa.mul %x, %rms_inv, %shift : (tensor<1x512x1024xf32>, tensor<1x512x1xf32>, tensor<1xi8>) -> tensor<1x512x1024xf32>
  %x_norm1_w = tosa.mul %x_norm1, %norm1_w, %shift : (tensor<1x512x1024xf32>, tensor<1x1x1024xf32>, tensor<1xi8>) -> tensor<1x512x1024xf32>

  // QKV projection
  %q_lin = tosa.matmul %x_norm1_w, %wq, %a_zp, %b_zp
      : (tensor<1x512x1024xf32>, tensor<1x1024x2048xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x512x2048xf32>
  %k_lin = tosa.matmul %x_norm1_w, %wk, %a_zp, %b_zp
      : (tensor<1x512x1024xf32>, tensor<1x1024x1024xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x512x1024xf32>
  %v_lin = tosa.matmul %x_norm1_w, %wv, %a_zp, %b_zp
      : (tensor<1x512x1024xf32>, tensor<1x1024x1024xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x512x1024xf32>

  // Q: [1,512,2048] -> [1,512,16,128] -> [1,16,512,128]
  // KV:[1,512,1024] -> [1,512,8,128]  -> [1,8,512,128]
  %s_q4_pre = tosa.const_shape {values = dense<[1, 512, 16, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %s_kv4_pre = tosa.const_shape {values = dense<[1, 512, 8, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %q4_pre = "tosa.reshape"(%q_lin, %s_q4_pre) : (tensor<1x512x2048xf32>, !tosa.shape<4>) -> tensor<1x512x16x128xf32>
  %k4_pre = "tosa.reshape"(%k_lin, %s_kv4_pre) : (tensor<1x512x1024xf32>, !tosa.shape<4>) -> tensor<1x512x8x128xf32>
  %v4_pre = "tosa.reshape"(%v_lin, %s_kv4_pre) : (tensor<1x512x1024xf32>, !tosa.shape<4>) -> tensor<1x512x8x128xf32>
  %q4 = tosa.transpose %q4_pre {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x512x16x128xf32>) -> tensor<1x16x512x128xf32>
  %k4 = tosa.transpose %k4_pre {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x512x8x128xf32>) -> tensor<1x8x512x128xf32>
  %v4 = tosa.transpose %v4_pre {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x512x8x128xf32>) -> tensor<1x8x512x128xf32>

  // q_norm / k_norm (RMSNorm on head_dim=128).
  %n_inv_128 = "tosa.const"() <{values = dense<7.812500e-03> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
  %eps4 = "tosa.const"() <{values = dense<1.0e-06> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
  %q_sq = tosa.mul %q4, %q4, %shift : (tensor<1x16x512x128xf32>, tensor<1x16x512x128xf32>, tensor<1xi8>) -> tensor<1x16x512x128xf32>
  %q_sum = tosa.reduce_sum %q_sq {axis = 3 : i32} : (tensor<1x16x512x128xf32>) -> tensor<1x16x512x1xf32>
  %q_mean = tosa.mul %q_sum, %n_inv_128, %shift : (tensor<1x16x512x1xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x16x512x1xf32>
  %q_mean_eps = tosa.add %q_mean, %eps4 : (tensor<1x16x512x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x16x512x1xf32>
  %q_rms_inv = tosa.rsqrt %q_mean_eps : (tensor<1x16x512x1xf32>) -> tensor<1x16x512x1xf32>
  %q_norm = tosa.mul %q4, %q_rms_inv, %shift : (tensor<1x16x512x128xf32>, tensor<1x16x512x1xf32>, tensor<1xi8>) -> tensor<1x16x512x128xf32>
  %q4n = tosa.mul %q_norm, %q_norm_w, %shift : (tensor<1x16x512x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x16x512x128xf32>

  %k_sq = tosa.mul %k4, %k4, %shift : (tensor<1x8x512x128xf32>, tensor<1x8x512x128xf32>, tensor<1xi8>) -> tensor<1x8x512x128xf32>
  %k_sum = tosa.reduce_sum %k_sq {axis = 3 : i32} : (tensor<1x8x512x128xf32>) -> tensor<1x8x512x1xf32>
  %k_mean = tosa.mul %k_sum, %n_inv_128, %shift : (tensor<1x8x512x1xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x8x512x1xf32>
  %k_mean_eps = tosa.add %k_mean, %eps4 : (tensor<1x8x512x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x8x512x1xf32>
  %k_rms_inv = tosa.rsqrt %k_mean_eps : (tensor<1x8x512x1xf32>) -> tensor<1x8x512x1xf32>
  %k_norm = tosa.mul %k4, %k_rms_inv, %shift : (tensor<1x8x512x128xf32>, tensor<1x8x512x1xf32>, tensor<1xi8>) -> tensor<1x8x512x128xf32>
  %k4n = tosa.mul %k_norm, %k_norm_w, %shift : (tensor<1x8x512x128xf32>, tensor<1x1x1x128xf32>, tensor<1xi8>) -> tensor<1x8x512x128xf32>

  // RoPE on Q/K (same formulation as decode path).
  %q_rope_empty = tensor.empty() : tensor<1x16x512x128xf32>
  %q_rope = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%q4n, %cos, %sin : tensor<1x16x512x128xf32>,
                         tensor<1x1x512x128xf32>,
                         tensor<1x1x512x128xf32>)
    outs(%q_rope_empty : tensor<1x16x512x128xf32>) {
  ^bb0(%q_val: f32, %cos_val: f32, %sin_val: f32, %out: f32):
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %d0 = linalg.index 0 : index
    %d1 = linalg.index 1 : index
    %d2 = linalg.index 2 : index
    %d3 = linalg.index 3 : index
    %partner = arith.addi %d3, %c64 : index
    %partner_mod = arith.remui %partner, %c128 : index
    %q_partner = tensor.extract %q4n[%d0, %d1, %d2, %partner_mod]
        : tensor<1x16x512x128xf32>
    %is_first_half = arith.cmpi ult, %d3, %c64 : index
    %neg_q_partner = arith.negf %q_partner : f32
    %rot_val = arith.select %is_first_half, %neg_q_partner, %q_partner : f32
    %q_cos = arith.mulf %q_val, %cos_val : f32
    %rot_sin = arith.mulf %rot_val, %sin_val : f32
    %result = arith.addf %q_cos, %rot_sin : f32
    linalg.yield %result : f32
  } -> tensor<1x16x512x128xf32>

  %k_rope_empty = tensor.empty() : tensor<1x8x512x128xf32>
  %k_rope = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%k4n, %cos, %sin : tensor<1x8x512x128xf32>,
                         tensor<1x1x512x128xf32>,
                         tensor<1x1x512x128xf32>)
    outs(%k_rope_empty : tensor<1x8x512x128xf32>) {
  ^bb0(%k_val: f32, %cos_val: f32, %sin_val: f32, %out: f32):
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %d0 = linalg.index 0 : index
    %d1 = linalg.index 1 : index
    %d2 = linalg.index 2 : index
    %d3 = linalg.index 3 : index
    %partner = arith.addi %d3, %c64 : index
    %partner_mod = arith.remui %partner, %c128 : index
    %k_partner = tensor.extract %k4n[%d0, %d1, %d2, %partner_mod]
        : tensor<1x8x512x128xf32>
    %is_first_half = arith.cmpi ult, %d3, %c64 : index
    %neg_k_partner = arith.negf %k_partner : f32
    %rot_val = arith.select %is_first_half, %neg_k_partner, %k_partner : f32
    %k_cos = arith.mulf %k_val, %cos_val : f32
    %rot_sin = arith.mulf %rot_val, %sin_val : f32
    %result = arith.addf %k_cos, %rot_sin : f32
    linalg.yield %result : f32
  } -> tensor<1x8x512x128xf32>

  // Q: [1, 16, 512, 128] -> [16, 512, 128], KV: [1, 8, 512, 128] -> [8, 512, 128]
  %s_q3 = tosa.const_shape {values = dense<[16, 512, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %s_kv3 = tosa.const_shape {values = dense<[8, 512, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %q3 = "tosa.reshape"(%q_rope, %s_q3) : (tensor<1x16x512x128xf32>, !tosa.shape<3>) -> tensor<16x512x128xf32>
  %k3 = "tosa.reshape"(%k_rope, %s_kv3) : (tensor<1x8x512x128xf32>, !tosa.shape<3>) -> tensor<8x512x128xf32>
  %v3 = "tosa.reshape"(%v4, %s_kv3) : (tensor<1x8x512x128xf32>, !tosa.shape<3>) -> tensor<8x512x128xf32>

  // Expand KV heads 8 -> 16: Q head i attends to KV head i//2.
  // Use linalg.generic instead of tosa.concat to avoid a GPU codegen bug where
  // the second half of the concatenated buffer is never filled (both copies go
  // to offset 0 of the destination).
  %k4_exp_empty = tensor.empty() : tensor<1x16x512x128xf32>
  %k4_exp = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 2, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%k_rope : tensor<1x8x512x128xf32>) outs(%k4_exp_empty : tensor<1x16x512x128xf32>) {
  ^bb0(%k_val: f32, %out: f32):
      linalg.yield %k_val : f32
  } -> tensor<1x16x512x128xf32>
  %v4_exp_empty = tensor.empty() : tensor<1x16x512x128xf32>
  %v4_exp = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 2, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%v4 : tensor<1x8x512x128xf32>) outs(%v4_exp_empty : tensor<1x16x512x128xf32>) {
  ^bb0(%v_val: f32, %out: f32):
      linalg.yield %v_val : f32
  } -> tensor<1x16x512x128xf32>
  %k3_exp = "tosa.reshape"(%k4_exp, %s_q3) : (tensor<1x16x512x128xf32>, !tosa.shape<3>) -> tensor<16x512x128xf32>
  %v3_exp = "tosa.reshape"(%v4_exp, %s_q3) : (tensor<1x16x512x128xf32>, !tosa.shape<3>) -> tensor<16x512x128xf32>

  // K^T for attention matmul.
  %k_t = tosa.transpose %k3_exp {perms = array<i32: 0, 2, 1>} : (tensor<16x512x128xf32>) -> tensor<16x128x512xf32>

  // scores = q @ k^T => [16, 512, 512]
  %scores = tosa.matmul %q3, %k_t, %a_zp, %b_zp
      : (tensor<16x512x128xf32>, tensor<16x128x512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<16x512x512xf32>

  // [16, 512, 512] -> [1, 16, 512, 512]
  %s_scores4 = tosa.const_shape {values = dense<[1, 16, 512, 512]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %scores4 = "tosa.reshape"(%scores, %s_scores4) : (tensor<16x512x512xf32>, !tosa.shape<4>) -> tensor<1x16x512x512xf32>

  // scale + causal mask
  %attn_scale = "tosa.const"() <{values = dense<8.8388348e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
  %scores_scaled = tosa.mul %scores4, %attn_scale, %shift
      : (tensor<1x16x512x512xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x16x512x512xf32>
  %scores_masked = tosa.add %scores_scaled, %attn_mask
      : (tensor<1x16x512x512xf32>, tensor<1x1x512x512xf32>) -> tensor<1x16x512x512xf32>

  // softmax(axis=3)
  %scores_max = tosa.reduce_max %scores_masked {axis = 3 : i32} : (tensor<1x16x512x512xf32>) -> tensor<1x16x512x1xf32>
  %scores_shifted = tosa.sub %scores_masked, %scores_max : (tensor<1x16x512x512xf32>, tensor<1x16x512x1xf32>) -> tensor<1x16x512x512xf32>
  %scores_exp = tosa.exp %scores_shifted : (tensor<1x16x512x512xf32>) -> tensor<1x16x512x512xf32>
  %scores_sum = tosa.reduce_sum %scores_exp {axis = 3 : i32} : (tensor<1x16x512x512xf32>) -> tensor<1x16x512x1xf32>
  %scores_sum_inv = tosa.reciprocal %scores_sum : (tensor<1x16x512x1xf32>) -> tensor<1x16x512x1xf32>
  %attn_prob4 = tosa.mul %scores_exp, %scores_sum_inv, %shift
      : (tensor<1x16x512x512xf32>, tensor<1x16x512x1xf32>, tensor<1xi8>) -> tensor<1x16x512x512xf32>

  // [1, 16, 512, 512] -> [16, 512, 512]
  %s_prob3 = tosa.const_shape {values = dense<[16, 512, 512]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %attn_prob3 = "tosa.reshape"(%attn_prob4, %s_prob3) : (tensor<1x16x512x512xf32>, !tosa.shape<3>) -> tensor<16x512x512xf32>

  // context = prob @ v -> [16, 512, 128]
  %ctx3 = tosa.matmul %attn_prob3, %v3_exp, %a_zp, %b_zp
      : (tensor<16x512x512xf32>, tensor<16x512x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<16x512x128xf32>

  // [16, 512, 128] -> [1, 16, 512, 128] -> transpose -> [1, 512, 16, 128] -> [1, 512, 2048]
  %s_ctx4 = tosa.const_shape {values = dense<[1, 16, 512, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %ctx4 = "tosa.reshape"(%ctx3, %s_ctx4) : (tensor<16x512x128xf32>, !tosa.shape<4>) -> tensor<1x16x512x128xf32>
  %ctx4_pre = tosa.transpose %ctx4 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x16x512x128xf32>) -> tensor<1x512x16x128xf32>
  %s_ctx_flat = tosa.const_shape {values = dense<[1, 512, 2048]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %ctx_flat = "tosa.reshape"(%ctx4_pre, %s_ctx_flat) : (tensor<1x512x16x128xf32>, !tosa.shape<3>) -> tensor<1x512x2048xf32>

  // output projection + residual
  %attn_proj = tosa.matmul %ctx_flat, %wo, %a_zp, %b_zp
      : (tensor<1x512x2048xf32>, tensor<1x2048x1024xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x512x1024xf32>
  %x_res1 = tosa.add %x, %attn_proj : (tensor<1x512x1024xf32>, tensor<1x512x1024xf32>) -> tensor<1x512x1024xf32>

  // RMSNorm #2
  %x2_sq = tosa.mul %x_res1, %x_res1, %shift : (tensor<1x512x1024xf32>, tensor<1x512x1024xf32>, tensor<1xi8>) -> tensor<1x512x1024xf32>
  %sum2_sq = tosa.reduce_sum %x2_sq {axis = 2 : i32} : (tensor<1x512x1024xf32>) -> tensor<1x512x1xf32>
  %mean2_sq = tosa.mul %sum2_sq, %n_inv, %shift : (tensor<1x512x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x512x1xf32>
  %mean2_sq_eps = tosa.add %mean2_sq, %eps : (tensor<1x512x1xf32>, tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
  %rms2_inv = tosa.rsqrt %mean2_sq_eps : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
  %x_norm2 = tosa.mul %x_res1, %rms2_inv, %shift : (tensor<1x512x1024xf32>, tensor<1x512x1xf32>, tensor<1xi8>) -> tensor<1x512x1024xf32>
  %x_norm2_w = tosa.mul %x_norm2, %norm2_w, %shift : (tensor<1x512x1024xf32>, tensor<1x1x1024xf32>, tensor<1xi8>) -> tensor<1x512x1024xf32>

  // SwiGLU FFN
  %gate = tosa.matmul %x_norm2_w, %ffn_gate, %a_zp, %b_zp
      : (tensor<1x512x1024xf32>, tensor<1x1024x3072xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x512x3072xf32>
  %up = tosa.matmul %x_norm2_w, %ffn_up, %a_zp, %b_zp
      : (tensor<1x512x1024xf32>, tensor<1x1024x3072xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x512x3072xf32>
  %sigmoid_gate = tosa.sigmoid %gate : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
  %silu_gate = tosa.mul %gate, %sigmoid_gate, %shift
      : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>, tensor<1xi8>) -> tensor<1x512x3072xf32>
  %ffn_hidden = tosa.mul %silu_gate, %up, %shift
      : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>, tensor<1xi8>) -> tensor<1x512x3072xf32>
  %ffn_out = tosa.matmul %ffn_hidden, %ffn_down, %a_zp, %b_zp
      : (tensor<1x512x3072xf32>, tensor<1x3072x1024xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x512x1024xf32>

  %output = tosa.add %x_res1, %ffn_out : (tensor<1x512x1024xf32>, tensor<1x512x1024xf32>) -> tensor<1x512x1024xf32>

  return %output, %k_rope, %v4 : tensor<1x512x1024xf32>, tensor<1x8x512x128xf32>, tensor<1x8x512x128xf32>
}
