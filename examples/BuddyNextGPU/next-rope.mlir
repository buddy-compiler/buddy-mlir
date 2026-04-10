func.func private @rtclock() -> f64

// Rotary Position Embedding (RoPE) for Qwen3-0.6B prefill
// num_heads = 32, seq_len = 40, head_dim = 128
// formula: output[i] = q[i]*cos[i] + rotate_half(q)[i]*sin[i]
//   rotate_half: for i < 64, rotate_half[i] = -q[i+64]
//                for i >= 64, rotate_half[i] =  q[i-64]
// Expressed as a single linalg.generic to avoid slice/concat.
func.func @kernel(
    %q   : tensor<1x32x40x128xf32>,
    %cos : tensor<1x1x40x128xf32>,
    %sin : tensor<1x1x40x128xf32>
) -> tensor<1x32x40x128xf32> {
  %empty = tensor.empty() : tensor<1x32x40x128xf32>
  %output = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, 0,  d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%q, %cos, %sin : tensor<1x32x40x128xf32>,
                         tensor<1x1x40x128xf32>,
                         tensor<1x1x40x128xf32>)
    outs(%empty : tensor<1x32x40x128xf32>) {
  ^bb0(%q_val: f32, %cos_val: f32, %sin_val: f32, %out: f32):
    %c64 = arith.constant 64 : index
    %d3  = linalg.index 3 : index
    // rotate_half partner index: (d3 + 64) % 128
    %partner = arith.addi %d3, %c64 : index
    %c128    = arith.constant 128 : index
    %partner_mod = arith.remui %partner, %c128 : index
    // load partner q value
    %d0 = linalg.index 0 : index
    %d1 = linalg.index 1 : index
    %d2 = linalg.index 2 : index
    %q_partner = tensor.extract %q[%d0, %d1, %d2, %partner_mod]
        : tensor<1x32x40x128xf32>
    // rotate_half sign: negative if d3 < 64
    %is_first_half = arith.cmpi ult, %d3, %c64 : index
    %neg_q_partner = arith.negf %q_partner : f32
    %rot_val = arith.select %is_first_half, %neg_q_partner, %q_partner : f32
    // output = q*cos + rotate_half*sin
    %q_cos    = arith.mulf %q_val,  %cos_val : f32
    %rot_sin  = arith.mulf %rot_val, %sin_val : f32
    %result   = arith.addf %q_cos, %rot_sin : f32
    linalg.yield %result : f32
  } -> tensor<1x32x40x128xf32>
  return %output : tensor<1x32x40x128xf32>
}

func.func @main() {
  %cst1 = arith.constant 1.0 : f32
  %cst2 = arith.constant 0.5 : f32

  %empty_q   = tensor.empty() : tensor<1x32x40x128xf32>
  %empty_cos = tensor.empty() : tensor<1x1x40x128xf32>
  %empty_sin = tensor.empty() : tensor<1x1x40x128xf32>

  %q   = linalg.fill ins(%cst1 : f32) outs(%empty_q   : tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %cos = linalg.fill ins(%cst2 : f32) outs(%empty_cos : tensor<1x1x40x128xf32>)  -> tensor<1x1x40x128xf32>
  %sin = linalg.fill ins(%cst2 : f32) outs(%empty_sin : tensor<1x1x40x128xf32>)  -> tensor<1x1x40x128xf32>

  %t_start = call @rtclock() : () -> f64
  %result = call @kernel(%q, %cos, %sin)
      : (tensor<1x32x40x128xf32>, tensor<1x1x40x128xf32>, tensor<1x1x40x128xf32>)
      -> tensor<1x32x40x128xf32>
  %t_end = call @rtclock() : () -> f64

  %tensor_unranked = tensor.cast %result : tensor<1x32x40x128xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
