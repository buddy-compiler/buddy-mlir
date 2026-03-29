func.func private @rtclock() -> f64

// RMSNorm for Qwen3-0.6B prefill
// hidden_size = 1536, seq_len = 40
// formula: output = x / rms(x) * weight,  rms(x) = sqrt(mean(x^2) + eps)
func.func @kernel(
    %x      : tensor<1x40x1536xf32>,
    %weight : tensor<1x1x1536xf32>
) -> tensor<1x40x1536xf32> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>

  // x^2: [1, 40, 1536]
  %x_sq = tosa.mul %x, %x, %shift
      : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>

  // sum(x^2) along hidden_size dim -> [1, 40, 1]
  %sum_sq = tosa.reduce_sum %x_sq {axis = 2 : i32}
      : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>

  // mean_sq = sum_sq / 1536  (1/1536 ≈ 6.510417e-04)
  %n_inv = "tosa.const"() <{values = dense<6.510417e-04> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %mean_sq = tosa.mul %sum_sq, %n_inv, %shift
      : (tensor<1x40x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x40x1xf32>

  // mean_sq + eps (1e-6)
  %eps = "tosa.const"() <{values = dense<1.0e-06> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %mean_sq_eps = tosa.add %mean_sq, %eps
      : (tensor<1x40x1xf32>, tensor<1x1x1xf32>) -> tensor<1x40x1xf32>

  // rms_inv = rsqrt(mean_sq + eps): [1, 40, 1]
  %rms_inv = tosa.rsqrt %mean_sq_eps
      : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>

  // x_norm = x * rms_inv  (broadcast [1,40,1] -> [1,40,1536])
  %x_norm = tosa.mul %x, %rms_inv, %shift
      : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>

  // output = x_norm * weight  (broadcast [1,1,1536] -> [1,40,1536])
  %output = tosa.mul %x_norm, %weight, %shift
      : (tensor<1x40x1536xf32>, tensor<1x1x1536xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>

  return %output : tensor<1x40x1536xf32>
}

func.func @main() {
  %cst1 = arith.constant 1.0 : f32
  %cstw = arith.constant 0.5 : f32

  %empty_x      = tensor.empty() : tensor<1x40x1536xf32>
  %empty_weight = tensor.empty() : tensor<1x1x1536xf32>

  %x      = linalg.fill ins(%cst1 : f32) outs(%empty_x      : tensor<1x40x1536xf32>)  -> tensor<1x40x1536xf32>
  %weight = linalg.fill ins(%cstw : f32) outs(%empty_weight : tensor<1x1x1536xf32>) -> tensor<1x1x1536xf32>

  %t_start = call @rtclock() : () -> f64
  %result = call @kernel(%x, %weight)
      : (tensor<1x40x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x40x1536xf32>
  %t_end = call @rtclock() : () -> f64

  %tensor_unranked = tensor.cast %result : tensor<1x40x1536xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
