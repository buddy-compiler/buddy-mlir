func.func private @rtclock() -> f64

// Qwen3-0.6B-like decode block (single layer, test-sized constants in main)
// hidden_size=1536, num_heads=12, head_dim=128, kv_seq_len=40
func.func @kernel(
    %x         : tensor<1x1x1536xf32>,
    %norm1_w   : tensor<1x1x1536xf32>,
    %wq        : tensor<1x1536x1536xf32>,
    %wk        : tensor<1x1536x1536xf32>,
    %wv        : tensor<1x1536x1536xf32>,
    %wo        : tensor<1x1536x1536xf32>,
    %attn_mask : tensor<1x1x1x40xf32>,
    %cos       : tensor<1x1x1x128xf32>,
    %sin       : tensor<1x1x1x128xf32>,
    %k_cache   : tensor<1x12x40x128xf32>,
    %v_cache   : tensor<1x12x40x128xf32>,
    %norm2_w   : tensor<1x1x1536xf32>,
    %ffn_gate  : tensor<1x1536x8960xf32>,
    %ffn_up    : tensor<1x1536x8960xf32>,
    %ffn_down  : tensor<1x8960x1536xf32>
) -> (tensor<1x1x1536xf32>, tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>

  // RMSNorm #1
  %x_sq = tosa.mul %x, %x, %shift : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1xi8>) -> tensor<1x1x1536xf32>
  %sum_sq = tosa.reduce_sum %x_sq {axis = 2 : i32} : (tensor<1x1x1536xf32>) -> tensor<1x1x1xf32>
  %n_inv = "tosa.const"() <{values = dense<6.510417e-04> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %mean_sq = tosa.mul %sum_sq, %n_inv, %shift : (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x1xf32>
  %eps = "tosa.const"() <{values = dense<1.0e-06> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %mean_sq_eps = tosa.add %mean_sq, %eps : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %rms_inv = tosa.rsqrt %mean_sq_eps : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %x_norm1 = tosa.mul %x, %rms_inv, %shift : (tensor<1x1x1536xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x1536xf32>
  %x_norm1_w = tosa.mul %x_norm1, %norm1_w, %shift : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1xi8>) -> tensor<1x1x1536xf32>

  // QKV projection
  %q_lin = tosa.matmul %x_norm1_w, %wq, %a_zp, %b_zp
      : (tensor<1x1x1536xf32>, tensor<1x1536x1536xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1536xf32>
  %k_lin = tosa.matmul %x_norm1_w, %wk, %a_zp, %b_zp
      : (tensor<1x1x1536xf32>, tensor<1x1536x1536xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1536xf32>
  %v_lin = tosa.matmul %x_norm1_w, %wv, %a_zp, %b_zp
      : (tensor<1x1x1536xf32>, tensor<1x1536x1536xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1536xf32>

  // [1, 1, 1536] -> [1, 1, 12, 128] -> transpose -> [1, 12, 1, 128]
  %s_head4_pre = tosa.const_shape {values = dense<[1, 1, 12, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %q4_pre = "tosa.reshape"(%q_lin, %s_head4_pre) : (tensor<1x1x1536xf32>, !tosa.shape<4>) -> tensor<1x1x12x128xf32>
  %k4_pre = "tosa.reshape"(%k_lin, %s_head4_pre) : (tensor<1x1x1536xf32>, !tosa.shape<4>) -> tensor<1x1x12x128xf32>
  %v4_pre = "tosa.reshape"(%v_lin, %s_head4_pre) : (tensor<1x1x1536xf32>, !tosa.shape<4>) -> tensor<1x1x12x128xf32>
  %q4 = tosa.transpose %q4_pre {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x1x12x128xf32>) -> tensor<1x12x1x128xf32>
  %k4 = tosa.transpose %k4_pre {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x1x12x128xf32>) -> tensor<1x12x1x128xf32>
  %v4 = tosa.transpose %v4_pre {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x1x12x128xf32>) -> tensor<1x12x1x128xf32>

  // RoPE on Q/K.
  %q_rope_empty = tensor.empty() : tensor<1x12x1x128xf32>
  %q_rope = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%q4, %cos, %sin : tensor<1x12x1x128xf32>,
                         tensor<1x1x1x128xf32>,
                         tensor<1x1x1x128xf32>)
    outs(%q_rope_empty : tensor<1x12x1x128xf32>) {
  ^bb0(%q_val: f32, %cos_val: f32, %sin_val: f32, %out: f32):
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %d0 = linalg.index 0 : index
    %d1 = linalg.index 1 : index
    %d2 = linalg.index 2 : index
    %d3 = linalg.index 3 : index
    %partner = arith.addi %d3, %c64 : index
    %partner_mod = arith.remui %partner, %c128 : index
    %q_partner = tensor.extract %q4[%d0, %d1, %d2, %partner_mod]
        : tensor<1x12x1x128xf32>
    %is_first_half = arith.cmpi ult, %d3, %c64 : index
    %neg_q_partner = arith.negf %q_partner : f32
    %rot_val = arith.select %is_first_half, %neg_q_partner, %q_partner : f32
    %q_cos = arith.mulf %q_val, %cos_val : f32
    %rot_sin = arith.mulf %rot_val, %sin_val : f32
    %result = arith.addf %q_cos, %rot_sin : f32
    linalg.yield %result : f32
  } -> tensor<1x12x1x128xf32>

  %k_rope_empty = tensor.empty() : tensor<1x12x1x128xf32>
  %k_rope = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%k4, %cos, %sin : tensor<1x12x1x128xf32>,
                         tensor<1x1x1x128xf32>,
                         tensor<1x1x1x128xf32>)
    outs(%k_rope_empty : tensor<1x12x1x128xf32>) {
  ^bb0(%k_val: f32, %cos_val: f32, %sin_val: f32, %out: f32):
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %d0 = linalg.index 0 : index
    %d1 = linalg.index 1 : index
    %d2 = linalg.index 2 : index
    %d3 = linalg.index 3 : index
    %partner = arith.addi %d3, %c64 : index
    %partner_mod = arith.remui %partner, %c128 : index
    %k_partner = tensor.extract %k4[%d0, %d1, %d2, %partner_mod]
        : tensor<1x12x1x128xf32>
    %is_first_half = arith.cmpi ult, %d3, %c64 : index
    %neg_k_partner = arith.negf %k_partner : f32
    %rot_val = arith.select %is_first_half, %neg_k_partner, %k_partner : f32
    %k_cos = arith.mulf %k_val, %cos_val : f32
    %rot_sin = arith.mulf %rot_val, %sin_val : f32
    %result = arith.addf %k_cos, %rot_sin : f32
    linalg.yield %result : f32
  } -> tensor<1x12x1x128xf32>

  // Update KV cache at decode position 10.
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %k_cache_updated = tensor.insert_slice %k_rope into %k_cache[%c0, %c0, %c10, %c0] [1, 12, 1, 128] [1, 1, 1, 1]
      : tensor<1x12x1x128xf32> into tensor<1x12x40x128xf32>
  %v_cache_updated = tensor.insert_slice %v4 into %v_cache[%c0, %c0, %c10, %c0] [1, 12, 1, 128] [1, 1, 1, 1]
      : tensor<1x12x1x128xf32> into tensor<1x12x40x128xf32>

  // [1, 12, 1, 128] -> [12, 1, 128]
  %s_q3 = tosa.const_shape {values = dense<[12, 1, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %q3 = "tosa.reshape"(%q_rope, %s_q3) : (tensor<1x12x1x128xf32>, !tosa.shape<3>) -> tensor<12x1x128xf32>

  // [1, 12, 40, 128] -> [12, 40, 128]
  %s_kv3 = tosa.const_shape {values = dense<[12, 40, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %k3 = "tosa.reshape"(%k_cache_updated, %s_kv3) : (tensor<1x12x40x128xf32>, !tosa.shape<3>) -> tensor<12x40x128xf32>
  %v3 = "tosa.reshape"(%v_cache_updated, %s_kv3) : (tensor<1x12x40x128xf32>, !tosa.shape<3>) -> tensor<12x40x128xf32>

  // scores = q @ k^T => [12, 1, 40]
  %k_t = tosa.transpose %k3 {perms = array<i32: 0, 2, 1>} : (tensor<12x40x128xf32>) -> tensor<12x128x40xf32>
  %scores = tosa.matmul %q3, %k_t, %a_zp, %b_zp
      : (tensor<12x1x128xf32>, tensor<12x128x40xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x1x40xf32>

  // [12, 1, 40] -> [1, 12, 1, 40]
  %s_scores4 = tosa.const_shape {values = dense<[1, 12, 1, 40]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %scores4 = "tosa.reshape"(%scores, %s_scores4) : (tensor<12x1x40xf32>, !tosa.shape<4>) -> tensor<1x12x1x40xf32>

  // scale + decode mask
  %attn_scale = "tosa.const"() <{values = dense<8.8388348e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
  %scores_scaled = tosa.mul %scores4, %attn_scale, %shift
      : (tensor<1x12x1x40xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x12x1x40xf32>
  %scores_masked = tosa.add %scores_scaled, %attn_mask
      : (tensor<1x12x1x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x12x1x40xf32>

  // softmax(axis=3)
  %scores_max = tosa.reduce_max %scores_masked {axis = 3 : i32} : (tensor<1x12x1x40xf32>) -> tensor<1x12x1x1xf32>
  %scores_shifted = tosa.sub %scores_masked, %scores_max : (tensor<1x12x1x40xf32>, tensor<1x12x1x1xf32>) -> tensor<1x12x1x40xf32>
  %scores_exp = tosa.exp %scores_shifted : (tensor<1x12x1x40xf32>) -> tensor<1x12x1x40xf32>
  %scores_sum = tosa.reduce_sum %scores_exp {axis = 3 : i32} : (tensor<1x12x1x40xf32>) -> tensor<1x12x1x1xf32>
  %scores_sum_inv = tosa.reciprocal %scores_sum : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x1xf32>
  %attn_prob4 = tosa.mul %scores_exp, %scores_sum_inv, %shift
      : (tensor<1x12x1x40xf32>, tensor<1x12x1x1xf32>, tensor<1xi8>) -> tensor<1x12x1x40xf32>

  // [1, 12, 1, 40] -> [12, 1, 40]
  %s_prob3 = tosa.const_shape {values = dense<[12, 1, 40]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %attn_prob3 = "tosa.reshape"(%attn_prob4, %s_prob3) : (tensor<1x12x1x40xf32>, !tosa.shape<3>) -> tensor<12x1x40xf32>

  // context = prob @ v -> [12, 1, 128]
  %ctx3 = tosa.matmul %attn_prob3, %v3, %a_zp, %b_zp
      : (tensor<12x1x40xf32>, tensor<12x40x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x1x128xf32>

  // [12, 1, 128] -> [1, 12, 1, 128] -> transpose -> [1, 1, 12, 128] -> [1, 1, 1536]
  %s_ctx4 = tosa.const_shape {values = dense<[1, 12, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %ctx4 = "tosa.reshape"(%ctx3, %s_ctx4) : (tensor<12x1x128xf32>, !tosa.shape<4>) -> tensor<1x12x1x128xf32>
  %ctx4_pre = tosa.transpose %ctx4 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x12x1x128xf32>) -> tensor<1x1x12x128xf32>
  %s_ctx_flat = tosa.const_shape {values = dense<[1, 1, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %ctx_flat = "tosa.reshape"(%ctx4_pre, %s_ctx_flat) : (tensor<1x1x12x128xf32>, !tosa.shape<3>) -> tensor<1x1x1536xf32>

  // output projection + residual
  %attn_proj = tosa.matmul %ctx_flat, %wo, %a_zp, %b_zp
      : (tensor<1x1x1536xf32>, tensor<1x1536x1536xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1536xf32>
  %x_res1 = tosa.add %x, %attn_proj : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>

  // RMSNorm #2
  %x2_sq = tosa.mul %x_res1, %x_res1, %shift : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1xi8>) -> tensor<1x1x1536xf32>
  %sum2_sq = tosa.reduce_sum %x2_sq {axis = 2 : i32} : (tensor<1x1x1536xf32>) -> tensor<1x1x1xf32>
  %mean2_sq = tosa.mul %sum2_sq, %n_inv, %shift : (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x1xf32>
  %mean2_sq_eps = tosa.add %mean2_sq, %eps : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %rms2_inv = tosa.rsqrt %mean2_sq_eps : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %x_norm2 = tosa.mul %x_res1, %rms2_inv, %shift : (tensor<1x1x1536xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x1536xf32>
  %x_norm2_w = tosa.mul %x_norm2, %norm2_w, %shift : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1xi8>) -> tensor<1x1x1536xf32>

  // SwiGLU FFN
  %gate = tosa.matmul %x_norm2_w, %ffn_gate, %a_zp, %b_zp
      : (tensor<1x1x1536xf32>, tensor<1x1536x8960xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x8960xf32>
  %up = tosa.matmul %x_norm2_w, %ffn_up, %a_zp, %b_zp
      : (tensor<1x1x1536xf32>, tensor<1x1536x8960xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x8960xf32>
  %sigmoid_gate = tosa.sigmoid %gate : (tensor<1x1x8960xf32>) -> tensor<1x1x8960xf32>
  %silu_gate = tosa.mul %gate, %sigmoid_gate, %shift
      : (tensor<1x1x8960xf32>, tensor<1x1x8960xf32>, tensor<1xi8>) -> tensor<1x1x8960xf32>
  %ffn_hidden = tosa.mul %silu_gate, %up, %shift
      : (tensor<1x1x8960xf32>, tensor<1x1x8960xf32>, tensor<1xi8>) -> tensor<1x1x8960xf32>
  %ffn_out = tosa.matmul %ffn_hidden, %ffn_down, %a_zp, %b_zp
      : (tensor<1x1x8960xf32>, tensor<1x8960x1536xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1536xf32>

  %output = tosa.add %x_res1, %ffn_out : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>

  return %output, %k_cache_updated, %v_cache_updated : tensor<1x1x1536xf32>, tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>
}

func.func @main() {
  %v_input = arith.constant 1.0 : f32
  %v_norm  = arith.constant 0.5 : f32
  %v_attn  = arith.constant 0.01 : f32
  %v_ffn   = arith.constant 0.01 : f32
  %v_cos   = arith.constant 1.0 : f32
  %v_sin   = arith.constant 0.0 : f32
  %v_zero  = arith.constant 0.0 : f32

  %e_input    = tensor.empty() : tensor<1x1x1536xf32>
  %e_norm1_w  = tensor.empty() : tensor<1x1x1536xf32>
  %e_wq       = tensor.empty() : tensor<1x1536x1536xf32>
  %e_wk       = tensor.empty() : tensor<1x1536x1536xf32>
  %e_wv       = tensor.empty() : tensor<1x1536x1536xf32>
  %e_wo       = tensor.empty() : tensor<1x1536x1536xf32>
  %e_mask     = tensor.empty() : tensor<1x1x1x40xf32>
  %e_cos      = tensor.empty() : tensor<1x1x1x128xf32>
  %e_sin      = tensor.empty() : tensor<1x1x1x128xf32>
  %e_k_cache  = tensor.empty() : tensor<1x12x40x128xf32>
  %e_v_cache  = tensor.empty() : tensor<1x12x40x128xf32>
  %e_norm2_w  = tensor.empty() : tensor<1x1x1536xf32>
  %e_ffn_gate = tensor.empty() : tensor<1x1536x8960xf32>
  %e_ffn_up   = tensor.empty() : tensor<1x1536x8960xf32>
  %e_ffn_down = tensor.empty() : tensor<1x8960x1536xf32>

  %input    = linalg.fill ins(%v_input : f32) outs(%e_input    : tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>
  %norm1_w  = linalg.fill ins(%v_norm  : f32) outs(%e_norm1_w  : tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>
  %wq       = linalg.fill ins(%v_attn  : f32) outs(%e_wq       : tensor<1x1536x1536xf32>) -> tensor<1x1536x1536xf32>
  %wk       = linalg.fill ins(%v_attn  : f32) outs(%e_wk       : tensor<1x1536x1536xf32>) -> tensor<1x1536x1536xf32>
  %wv       = linalg.fill ins(%v_attn  : f32) outs(%e_wv       : tensor<1x1536x1536xf32>) -> tensor<1x1536x1536xf32>
  %wo       = linalg.fill ins(%v_attn  : f32) outs(%e_wo       : tensor<1x1536x1536xf32>) -> tensor<1x1536x1536xf32>
  %cos      = linalg.fill ins(%v_cos   : f32) outs(%e_cos      : tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
  %sin      = linalg.fill ins(%v_sin   : f32) outs(%e_sin      : tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
  %k_cache  = linalg.fill ins(%v_zero  : f32) outs(%e_k_cache  : tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
  %v_cache  = linalg.fill ins(%v_zero  : f32) outs(%e_v_cache  : tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
  %norm2_w  = linalg.fill ins(%v_norm  : f32) outs(%e_norm2_w  : tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>
  %ffn_gate = linalg.fill ins(%v_ffn   : f32) outs(%e_ffn_gate : tensor<1x1536x8960xf32>) -> tensor<1x1536x8960xf32>
  %ffn_up   = linalg.fill ins(%v_ffn   : f32) outs(%e_ffn_up   : tensor<1x1536x8960xf32>) -> tensor<1x1536x8960xf32>
  %ffn_down = linalg.fill ins(%v_ffn   : f32) outs(%e_ffn_down : tensor<1x8960x1536xf32>) -> tensor<1x8960x1536xf32>

  // temporary stable path: zero mask for decode.
  %mask = linalg.fill ins(%v_zero : f32) outs(%e_mask : tensor<1x1x1x40xf32>) -> tensor<1x1x1x40xf32>

  %t_start = call @rtclock() : () -> f64
  %hidden, %k_out, %v_out = call @kernel(%input, %norm1_w, %wq, %wk, %wv, %wo, %mask, %cos, %sin, %k_cache, %v_cache, %norm2_w, %ffn_gate, %ffn_up, %ffn_down)
      : (tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x1536x1536xf32>, tensor<1x1536x1536xf32>, tensor<1x1536x1536xf32>, tensor<1x1536x1536xf32>, tensor<1x1x1x40xf32>, tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>, tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>, tensor<1x1x1536xf32>, tensor<1x1536x8960xf32>, tensor<1x1536x8960xf32>, tensor<1x8960x1536xf32>)
      -> (tensor<1x1x1536xf32>, tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>)
  %t_end = call @rtclock() : () -> f64

  // Scalar checkpoints for CPU/GPU numeric comparison.
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %h000 = tensor.extract %hidden[%c0, %c0, %c0] : tensor<1x1x1536xf32>
  %k0100 = tensor.extract %k_out[%c0, %c0, %c10, %c0] : tensor<1x12x40x128xf32>
  %v0100 = tensor.extract %v_out[%c0, %c0, %c10, %c0] : tensor<1x12x40x128xf32>
  vector.print %h000 : f32
  vector.print %k0100 : f32
  vector.print %v0100 : f32

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
