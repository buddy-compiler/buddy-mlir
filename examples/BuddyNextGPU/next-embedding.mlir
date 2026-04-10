func.func private @rtclock() -> f64

// Token Embedding lookup for Qwen3-0.6B prefill
// vocab_size = 512 (test), hidden_size = 1536, seq_len = 40
// tosa.gather: values[1, vocab, hidden], indices[1, seq] -> output[1, seq, hidden]
func.func @kernel(
    %table   : tensor<1x512x1536xf32>,
    %indices : tensor<1x40xi32>
) -> tensor<1x40x1536xf32> {
  %output = tosa.gather %table, %indices
      : (tensor<1x512x1536xf32>, tensor<1x40xi32>) -> tensor<1x40x1536xf32>
  return %output : tensor<1x40x1536xf32>
}

func.func @main() {
  %cst = arith.constant 1.0 : f32

  %empty_table = tensor.empty() : tensor<1x512x1536xf32>
  %table   = linalg.fill ins(%cst : f32) outs(%empty_table : tensor<1x512x1536xf32>) -> tensor<1x512x1536xf32>
  // all tokens look up index 0
  %indices = arith.constant dense<0> : tensor<1x40xi32>

  %t_start = call @rtclock() : () -> f64
  %result = call @kernel(%table, %indices)
      : (tensor<1x512x1536xf32>, tensor<1x40xi32>) -> tensor<1x40x1536xf32>
  %t_end = call @rtclock() : () -> f64

  %tensor_unranked = tensor.cast %result : tensor<1x40x1536xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
