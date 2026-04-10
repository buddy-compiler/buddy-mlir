func.func private @rtclock() -> f64

// SwiGLU FFN for Qwen3-0.6B prefill
// hidden_size = 1536, intermediate_size = 8960, seq_len = 40
func.func @kernel(
    %input   : tensor<1x40x1536xf32>,
    %gate_w  : tensor<1x1536x8960xf32>,
    %up_w    : tensor<1x1536x8960xf32>,
    %down_w  : tensor<1x8960x1536xf32>
) -> tensor<1x40x1536xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>

  // gate = input @ gate_w  [1, 40, 8960]
  %gate = tosa.matmul %input, %gate_w, %a_zp, %b_zp
      : (tensor<1x40x1536xf32>, tensor<1x1536x8960xf32>, tensor<1xf32>, tensor<1xf32>)
      -> tensor<1x40x8960xf32>

  // up = input @ up_w  [1, 40, 8960]
  %up = tosa.matmul %input, %up_w, %a_zp, %b_zp
      : (tensor<1x40x1536xf32>, tensor<1x1536x8960xf32>, tensor<1xf32>, tensor<1xf32>)
      -> tensor<1x40x8960xf32>

  // silu(gate) = gate * sigmoid(gate)
  %sigmoid_gate = tosa.sigmoid %gate : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
  %silu_gate = tosa.mul %gate, %sigmoid_gate, %shift
      : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>, tensor<1xi8>) -> tensor<1x40x8960xf32>

  // hidden = silu(gate) * up
  %hidden = tosa.mul %silu_gate, %up, %shift
      : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>, tensor<1xi8>) -> tensor<1x40x8960xf32>

  // output = hidden @ down_w  [1, 40, 1536]
  %output = tosa.matmul %hidden, %down_w, %a_zp, %b_zp
      : (tensor<1x40x8960xf32>, tensor<1x8960x1536xf32>, tensor<1xf32>, tensor<1xf32>)
      -> tensor<1x40x1536xf32>

  return %output : tensor<1x40x1536xf32>
}

func.func @main() {
  %cst1 = arith.constant 1.0 : f32
  %cst2 = arith.constant 0.01 : f32

  %empty_input  = tensor.empty() : tensor<1x40x1536xf32>
  %empty_gate_w = tensor.empty() : tensor<1x1536x8960xf32>
  %empty_up_w   = tensor.empty() : tensor<1x1536x8960xf32>
  %empty_down_w = tensor.empty() : tensor<1x8960x1536xf32>

  %input  = linalg.fill ins(%cst1 : f32) outs(%empty_input  : tensor<1x40x1536xf32>)  -> tensor<1x40x1536xf32>
  %gate_w = linalg.fill ins(%cst2 : f32) outs(%empty_gate_w : tensor<1x1536x8960xf32>) -> tensor<1x1536x8960xf32>
  %up_w   = linalg.fill ins(%cst2 : f32) outs(%empty_up_w   : tensor<1x1536x8960xf32>) -> tensor<1x1536x8960xf32>
  %down_w = linalg.fill ins(%cst2 : f32) outs(%empty_down_w : tensor<1x8960x1536xf32>) -> tensor<1x8960x1536xf32>

  %t_start = call @rtclock() : () -> f64
  %result = call @kernel(%input, %gate_w, %up_w, %down_w)
      : (tensor<1x40x1536xf32>, tensor<1x1536x8960xf32>, tensor<1x1536x8960xf32>, tensor<1x8960x1536xf32>)
      -> tensor<1x40x1536xf32>
  %t_end = call @rtclock() : () -> f64

  %tensor_unranked = tensor.cast %result : tensor<1x40x1536xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
  // Print timings.

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
