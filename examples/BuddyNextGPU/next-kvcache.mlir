func.func private @rtclock() -> f64

// KV Cache update for decode phase (Qwen3-0.6B)
// num_kv_heads = 32, max_seq_len = 40, head_dim = 128
//
// Each decode step appends one new K/V token to the cache:
//   k_cache[b, h, pos, :] = new_k[b, h, 0, :]
//
// Expressed as a parallel linalg.generic (GPU-friendly):
//   output[b, h, seq, d] = new_k[b, h, 0, d]  if seq == pos
//                         = k_cache[b, h, seq, d]  otherwise
func.func @kernel(
    %k_cache : tensor<1x32x40x128xf32>,
    %v_cache : tensor<1x32x40x128xf32>,
    %new_k   : tensor<1x32x1x128xf32>,
    %new_v   : tensor<1x32x1x128xf32>
) -> (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>) {
  %empty_k = tensor.empty() : tensor<1x32x40x128xf32>
  %empty_v = tensor.empty() : tensor<1x32x40x128xf32>

  // Insert new_k at decode position 10
  %k_updated = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, 0,  d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%k_cache, %new_k : tensor<1x32x40x128xf32>, tensor<1x32x1x128xf32>)
    outs(%empty_k : tensor<1x32x40x128xf32>) {
  ^bb0(%cache_val: f32, %new_val: f32, %out: f32):
    %pos   = arith.constant 10 : index
    %seq   = linalg.index 2 : index
    %at_pos = arith.cmpi eq, %seq, %pos : index
    %result = arith.select %at_pos, %new_val, %cache_val : f32
    linalg.yield %result : f32
  } -> tensor<1x32x40x128xf32>

  // Insert new_v at the same position
  %v_updated = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, 0,  d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%v_cache, %new_v : tensor<1x32x40x128xf32>, tensor<1x32x1x128xf32>)
    outs(%empty_v : tensor<1x32x40x128xf32>) {
  ^bb0(%cache_val: f32, %new_val: f32, %out: f32):
    %pos    = arith.constant 10 : index
    %seq    = linalg.index 2 : index
    %at_pos = arith.cmpi eq, %seq, %pos : index
    %result = arith.select %at_pos, %new_val, %cache_val : f32
    linalg.yield %result : f32
  } -> tensor<1x32x40x128xf32>

  return %k_updated, %v_updated : tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>
}

func.func @main() {
  %zero = arith.constant 0.0 : f32
  %one  = arith.constant 1.0 : f32

  %empty_cache = tensor.empty() : tensor<1x32x40x128xf32>
  %empty_new   = tensor.empty() : tensor<1x32x1x128xf32>

  // KV cache initialized to zeros, new token filled with 1.0
  %k_cache = linalg.fill ins(%zero : f32) outs(%empty_cache : tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %v_cache = linalg.fill ins(%zero : f32) outs(%empty_cache : tensor<1x32x40x128xf32>) -> tensor<1x32x40x128xf32>
  %new_k   = linalg.fill ins(%one  : f32) outs(%empty_new   : tensor<1x32x1x128xf32>)  -> tensor<1x32x1x128xf32>
  %new_v   = linalg.fill ins(%one  : f32) outs(%empty_new   : tensor<1x32x1x128xf32>)  -> tensor<1x32x1x128xf32>

  %t_start = call @rtclock() : () -> f64
  %k_out, %v_out = call @kernel(%k_cache, %v_cache, %new_k, %new_v)
      : (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>,
         tensor<1x32x1x128xf32>,  tensor<1x32x1x128xf32>)
      -> (tensor<1x32x40x128xf32>, tensor<1x32x40x128xf32>)
  %t_end = call @rtclock() : () -> f64

  %tensor_unranked = tensor.cast %k_out : tensor<1x32x40x128xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
