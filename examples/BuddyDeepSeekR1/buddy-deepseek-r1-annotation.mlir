#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  // ===========================================================================
  // DeepSeek R1 Transformer Model - MLIR Representation
  // ===========================================================================
  // This file contains the MLIR representation of DeepSeek R1 transformer model
  //
  // Model Architecture Summary:
  // - Vocabulary Size: 151,936 tokens
  // - Hidden Dimension: 1,536
  // - Attention Heads: 12
  // - Head Dimension: 128 (1536/12)
  // - Transformer Layers: 28 layers
  // - FFN Expansion Ratio: ~5.83 (8960/1536)
  // - Sequence Length: 40 tokens (in this example)
  // - Activation: SwiGLU (Sigmoid-Gated Linear Unit)
  // - Positional Encoding: RoPE (Rotary Position Embedding)
  // - Normalization: Layer Normalization (Pre-Norm)
  //
  // Main Components:
  // 1. Token Embedding: %arg0 (151936x1536) - vocabulary_size x hidden_dim
  // 2. Positional Encoding: %arg3 (64) - rotary embedding frequencies
  // 3. Transformer Layers (28 layers, each with):
  //    - Multi-Head Self-Attention (12 heads, 128 dims per head)
  //    - SwiGLU Feed-Forward Network (1536 -> 8960 -> 1536)
  //    - Residual Connections and Layer Normalization
  // 4. Output Projection: %arg341 (151936x1536) - hidden_dim x vocabulary_size
  //
  // Note: This MLIR file represents a single forward pass through the model.
  // The model follows the standard Transformer decoder architecture with
  // causal attention masking for autoregressive language modeling.
  // ===========================================

  func.func @subgraph0(%arg0: tensor<151936x1536xf32>, %arg1: tensor<1x40xi64>, %arg2: tensor<1x40xi64>, %arg3: tensor<64xf32>, %arg4: tensor<1536xf32>, %arg5: tensor<1536x1536xf32>, %arg6: tensor<1536xf32>, %arg7: tensor<256x1536xf32>, %arg8: tensor<256xf32>, %arg9: tensor<256x1536xf32>, %arg10: tensor<256xf32>, %arg11: tensor<1536x1536xf32>, %arg12: tensor<1536xf32>, %arg13: tensor<8960x1536xf32>, %arg14: tensor<8960x1536xf32>, %arg15: tensor<1536x8960xf32>, %arg16: tensor<1536xf32>, %arg17: tensor<1536x1536xf32>, %arg18: tensor<1536xf32>, %arg19: tensor<256x1536xf32>, %arg20: tensor<256xf32>, %arg21: tensor<256x1536xf32>, %arg22: tensor<256xf32>, %arg23: tensor<1536x1536xf32>, %arg24: tensor<1536xf32>, %arg25: tensor<8960x1536xf32>, %arg26: tensor<8960x1536xf32>, %arg27: tensor<1536x8960xf32>, %arg28: tensor<1536xf32>, %arg29: tensor<1536x1536xf32>, %arg30: tensor<1536xf32>, %arg31: tensor<256x1536xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256x1536xf32>, %arg34: tensor<256xf32>, %arg35: tensor<1536x1536xf32>, %arg36: tensor<1536xf32>, %arg37: tensor<8960x1536xf32>, %arg38: tensor<8960x1536xf32>, %arg39: tensor<1536x8960xf32>, %arg40: tensor<1536xf32>, %arg41: tensor<1536x1536xf32>, %arg42: tensor<1536xf32>, %arg43: tensor<256x1536xf32>, %arg44: tensor<256xf32>, %arg45: tensor<256x1536xf32>, %arg46: tensor<256xf32>, %arg47: tensor<1536x1536xf32>, %arg48: tensor<1536xf32>, %arg49: tensor<8960x1536xf32>, %arg50: tensor<8960x1536xf32>, %arg51: tensor<1536x8960xf32>, %arg52: tensor<1536xf32>, %arg53: tensor<1536x1536xf32>, %arg54: tensor<1536xf32>, %arg55: tensor<256x1536xf32>, %arg56: tensor<256xf32>, %arg57: tensor<256x1536xf32>, %arg58: tensor<256xf32>, %arg59: tensor<1536x1536xf32>, %arg60: tensor<1536xf32>, %arg61: tensor<8960x1536xf32>, %arg62: tensor<8960x1536xf32>, %arg63: tensor<1536x8960xf32>, %arg64: tensor<1536xf32>, %arg65: tensor<1536x1536xf32>, %arg66: tensor<1536xf32>, %arg67: tensor<256x1536xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256x1536xf32>, %arg70: tensor<256xf32>, %arg71: tensor<1536x1536xf32>, %arg72: tensor<1536xf32>, %arg73: tensor<8960x1536xf32>, %arg74: tensor<8960x1536xf32>, %arg75: tensor<1536x8960xf32>, %arg76: tensor<1536xf32>, %arg77: tensor<1536x1536xf32>, %arg78: tensor<1536xf32>, %arg79: tensor<256x1536xf32>, %arg80: tensor<256xf32>, %arg81: tensor<256x1536xf32>, %arg82: tensor<256xf32>, %arg83: tensor<1536x1536xf32>, %arg84: tensor<1536xf32>, %arg85: tensor<8960x1536xf32>, %arg86: tensor<8960x1536xf32>, %arg87: tensor<1536x8960xf32>, %arg88: tensor<1536xf32>, %arg89: tensor<1536x1536xf32>, %arg90: tensor<1536xf32>, %arg91: tensor<256x1536xf32>, %arg92: tensor<256xf32>, %arg93: tensor<256x1536xf32>, %arg94: tensor<256xf32>, %arg95: tensor<1536x1536xf32>, %arg96: tensor<1536xf32>, %arg97: tensor<8960x1536xf32>, %arg98: tensor<8960x1536xf32>, %arg99: tensor<1536x8960xf32>, %arg100: tensor<1536xf32>, %arg101: tensor<1536x1536xf32>, %arg102: tensor<1536xf32>, %arg103: tensor<256x1536xf32>, %arg104: tensor<256xf32>, %arg105: tensor<256x1536xf32>, %arg106: tensor<256xf32>, %arg107: tensor<1536x1536xf32>, %arg108: tensor<1536xf32>, %arg109: tensor<8960x1536xf32>, %arg110: tensor<8960x1536xf32>, %arg111: tensor<1536x8960xf32>, %arg112: tensor<1536xf32>, %arg113: tensor<1536x1536xf32>, %arg114: tensor<1536xf32>, %arg115: tensor<256x1536xf32>, %arg116: tensor<256xf32>, %arg117: tensor<256x1536xf32>, %arg118: tensor<256xf32>, %arg119: tensor<1536x1536xf32>, %arg120: tensor<1536xf32>, %arg121: tensor<8960x1536xf32>, %arg122: tensor<8960x1536xf32>, %arg123: tensor<1536x8960xf32>, %arg124: tensor<1536xf32>, %arg125: tensor<1536x1536xf32>, %arg126: tensor<1536xf32>, %arg127: tensor<256x1536xf32>, %arg128: tensor<256xf32>, %arg129: tensor<256x1536xf32>, %arg130: tensor<256xf32>, %arg131: tensor<1536x1536xf32>, %arg132: tensor<1536xf32>, %arg133: tensor<8960x1536xf32>, %arg134: tensor<8960x1536xf32>, %arg135: tensor<1536x8960xf32>, %arg136: tensor<1536xf32>, %arg137: tensor<1536x1536xf32>, %arg138: tensor<1536xf32>, %arg139: tensor<256x1536xf32>, %arg140: tensor<256xf32>, %arg141: tensor<256x1536xf32>, %arg142: tensor<256xf32>, %arg143: tensor<1536x1536xf32>, %arg144: tensor<1536xf32>, %arg145: tensor<8960x1536xf32>, %arg146: tensor<8960x1536xf32>, %arg147: tensor<1536x8960xf32>, %arg148: tensor<1536xf32>, %arg149: tensor<1536x1536xf32>, %arg150: tensor<1536xf32>, %arg151: tensor<256x1536xf32>, %arg152: tensor<256xf32>, %arg153: tensor<256x1536xf32>, %arg154: tensor<256xf32>, %arg155: tensor<1536x1536xf32>, %arg156: tensor<1536xf32>, %arg157: tensor<8960x1536xf32>, %arg158: tensor<8960x1536xf32>, %arg159: tensor<1536x8960xf32>, %arg160: tensor<1536xf32>, %arg161: tensor<1536x1536xf32>, %arg162: tensor<1536xf32>, %arg163: tensor<256x1536xf32>, %arg164: tensor<256xf32>, %arg165: tensor<256x1536xf32>, %arg166: tensor<256xf32>, %arg167: tensor<1536x1536xf32>, %arg168: tensor<1536xf32>, %arg169: tensor<8960x1536xf32>, %arg170: tensor<8960x1536xf32>, %arg171: tensor<1536x8960xf32>, %arg172: tensor<1536xf32>, %arg173: tensor<1536x1536xf32>, %arg174: tensor<1536xf32>, %arg175: tensor<256x1536xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256x1536xf32>, %arg178: tensor<256xf32>, %arg179: tensor<1536x1536xf32>, %arg180: tensor<1536xf32>, %arg181: tensor<8960x1536xf32>, %arg182: tensor<8960x1536xf32>, %arg183: tensor<1536x8960xf32>, %arg184: tensor<1536xf32>, %arg185: tensor<1536x1536xf32>, %arg186: tensor<1536xf32>, %arg187: tensor<256x1536xf32>, %arg188: tensor<256xf32>, %arg189: tensor<256x1536xf32>, %arg190: tensor<256xf32>, %arg191: tensor<1536x1536xf32>, %arg192: tensor<1536xf32>, %arg193: tensor<8960x1536xf32>, %arg194: tensor<8960x1536xf32>, %arg195: tensor<1536x8960xf32>, %arg196: tensor<1536xf32>, %arg197: tensor<1536x1536xf32>, %arg198: tensor<1536xf32>, %arg199: tensor<256x1536xf32>, %arg200: tensor<256xf32>, %arg201: tensor<256x1536xf32>, %arg202: tensor<256xf32>, %arg203: tensor<1536x1536xf32>, %arg204: tensor<1536xf32>, %arg205: tensor<8960x1536xf32>, %arg206: tensor<8960x1536xf32>, %arg207: tensor<1536x8960xf32>, %arg208: tensor<1536xf32>, %arg209: tensor<1536x1536xf32>, %arg210: tensor<1536xf32>, %arg211: tensor<256x1536xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256x1536xf32>, %arg214: tensor<256xf32>, %arg215: tensor<1536x1536xf32>, %arg216: tensor<1536xf32>, %arg217: tensor<8960x1536xf32>, %arg218: tensor<8960x1536xf32>, %arg219: tensor<1536x8960xf32>, %arg220: tensor<1536xf32>, %arg221: tensor<1536x1536xf32>, %arg222: tensor<1536xf32>, %arg223: tensor<256x1536xf32>, %arg224: tensor<256xf32>, %arg225: tensor<256x1536xf32>, %arg226: tensor<256xf32>, %arg227: tensor<1536x1536xf32>, %arg228: tensor<1536xf32>, %arg229: tensor<8960x1536xf32>, %arg230: tensor<8960x1536xf32>, %arg231: tensor<1536x8960xf32>, %arg232: tensor<1536xf32>, %arg233: tensor<1536x1536xf32>, %arg234: tensor<1536xf32>, %arg235: tensor<256x1536xf32>, %arg236: tensor<256xf32>, %arg237: tensor<256x1536xf32>, %arg238: tensor<256xf32>, %arg239: tensor<1536x1536xf32>, %arg240: tensor<1536xf32>, %arg241: tensor<8960x1536xf32>, %arg242: tensor<8960x1536xf32>, %arg243: tensor<1536x8960xf32>, %arg244: tensor<1536xf32>, %arg245: tensor<1536x1536xf32>, %arg246: tensor<1536xf32>, %arg247: tensor<256x1536xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256x1536xf32>, %arg250: tensor<256xf32>, %arg251: tensor<1536x1536xf32>, %arg252: tensor<1536xf32>, %arg253: tensor<8960x1536xf32>, %arg254: tensor<8960x1536xf32>, %arg255: tensor<1536x8960xf32>, %arg256: tensor<1536xf32>, %arg257: tensor<1536x1536xf32>, %arg258: tensor<1536xf32>, %arg259: tensor<256x1536xf32>, %arg260: tensor<256xf32>, %arg261: tensor<256x1536xf32>, %arg262: tensor<256xf32>, %arg263: tensor<1536x1536xf32>, %arg264: tensor<1536xf32>, %arg265: tensor<8960x1536xf32>, %arg266: tensor<8960x1536xf32>, %arg267: tensor<1536x8960xf32>, %arg268: tensor<1536xf32>, %arg269: tensor<1536x1536xf32>, %arg270: tensor<1536xf32>, %arg271: tensor<256x1536xf32>, %arg272: tensor<256xf32>, %arg273: tensor<256x1536xf32>, %arg274: tensor<256xf32>, %arg275: tensor<1536x1536xf32>, %arg276: tensor<1536xf32>, %arg277: tensor<8960x1536xf32>, %arg278: tensor<8960x1536xf32>, %arg279: tensor<1536x8960xf32>, %arg280: tensor<1536xf32>, %arg281: tensor<1536x1536xf32>, %arg282: tensor<1536xf32>, %arg283: tensor<256x1536xf32>, %arg284: tensor<256xf32>, %arg285: tensor<256x1536xf32>, %arg286: tensor<256xf32>, %arg287: tensor<1536x1536xf32>, %arg288: tensor<1536xf32>, %arg289: tensor<8960x1536xf32>, %arg290: tensor<8960x1536xf32>, %arg291: tensor<1536x8960xf32>, %arg292: tensor<1536xf32>, %arg293: tensor<1536x1536xf32>, %arg294: tensor<1536xf32>, %arg295: tensor<256x1536xf32>, %arg296: tensor<256xf32>, %arg297: tensor<256x1536xf32>, %arg298: tensor<256xf32>, %arg299: tensor<1536x1536xf32>, %arg300: tensor<1536xf32>, %arg301: tensor<8960x1536xf32>, %arg302: tensor<8960x1536xf32>, %arg303: tensor<1536x8960xf32>, %arg304: tensor<1536xf32>, %arg305: tensor<1536x1536xf32>, %arg306: tensor<1536xf32>, %arg307: tensor<256x1536xf32>, %arg308: tensor<256xf32>, %arg309: tensor<256x1536xf32>, %arg310: tensor<256xf32>, %arg311: tensor<1536x1536xf32>, %arg312: tensor<1536xf32>, %arg313: tensor<8960x1536xf32>, %arg314: tensor<8960x1536xf32>, %arg315: tensor<1536x8960xf32>, %arg316: tensor<1536xf32>, %arg317: tensor<1536x1536xf32>, %arg318: tensor<1536xf32>, %arg319: tensor<256x1536xf32>, %arg320: tensor<256xf32>, %arg321: tensor<256x1536xf32>, %arg322: tensor<256xf32>, %arg323: tensor<1536x1536xf32>, %arg324: tensor<1536xf32>, %arg325: tensor<8960x1536xf32>, %arg326: tensor<8960x1536xf32>, %arg327: tensor<1536x8960xf32>, %arg328: tensor<1536xf32>, %arg329: tensor<1536x1536xf32>, %arg330: tensor<1536xf32>, %arg331: tensor<256x1536xf32>, %arg332: tensor<256xf32>, %arg333: tensor<256x1536xf32>, %arg334: tensor<256xf32>, %arg335: tensor<1536x1536xf32>, %arg336: tensor<1536xf32>, %arg337: tensor<8960x1536xf32>, %arg338: tensor<8960x1536xf32>, %arg339: tensor<1536x8960xf32>, %arg340: tensor<1536xf32>, %arg341: tensor<151936x1536xf32>) -> tensor<1x40x151936xf32> {
    // ===========================================
    // 1. TOKEN EMBEDDING LAYER
    // ===========================================
    // Input: %arg1 (1x40xi64) - token IDs for sequence of length 40
    // Embedding weights: %arg0 (151936x1536) - vocabulary_size x hidden_dim
    // Output: token embeddings (1x40x1536)

    %0 = tosa.cast %arg1 : (tensor<1x40xi64>) -> tensor<1x40xi32>
    %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 151936, 1536>} : (tensor<151936x1536xf32>) -> tensor<1x151936x1536xf32>
    %2 = tosa.gather %1, %0 : (tensor<1x151936x1536xf32>, tensor<1x40xi32>) -> tensor<1x40x1536xf32>
    %3 = tosa.reshape %2 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    // ===========================================
    // 2. POSITIONAL ENCODING & ATTENTION MASK
    // ===========================================
    // Create causal attention mask to prevent attending to future tokens
    // Input: %arg2 (1x40xi64) - position indices for causal masking
    // Rotary embedding weights: %arg3 (64xf32) - rotary embedding frequencies

    %4 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    %5 = tosa.reshape %4 {new_shape = array<i64: 1, 40>} : (tensor<40xi64>) -> tensor<1x40xi64>
    %cst = arith.constant dense<-3.40282347E+38> : tensor<40x40xf32>
    %6 = "tosa.const"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi64>}> : () -> tensor<40xi64>
    %7 = tosa.reshape %4 {new_shape = array<i64: 40, 1>} : (tensor<40xi64>) -> tensor<40x1xi64>
    %8 = tensor.empty() : tensor<40x40xi1>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6, %7 : tensor<40xi64>, tensor<40x1xi64>) outs(%8 : tensor<40x40xi1>) {
    ^bb0(%in: i64, %in_843: i64, %out: i1):
      %3964 = arith.cmpi sgt, %in, %in_843 : i64
      linalg.yield %3964 : i1
    } -> tensor<40x40xi1>
    %10 = tosa.cast %9 : (tensor<40x40xi1>) -> tensor<40x40xf32>
    %11 = tosa.mul %cst, %10 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %12 = tosa.reshape %11 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %13 = tosa.reshape %12 {new_shape = array<i64: 1, 1, 40, 40>} : (tensor<1x40x40xf32>) -> tensor<1x1x40x40xf32>
    %extracted_slice = tensor.extract_slice %13[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_0 = tensor.extract_slice %extracted_slice[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %14 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40x40xf32>}> : () -> tensor<1x1x40x40xf32>
    %15 = tosa.add %extracted_slice_0, %14 : (tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    %16 = tosa.identity %15 : (tensor<1x1x40x40xf32>) -> tensor<1x1x40x40xf32>
    %extracted_slice_1 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_2 = tensor.extract_slice %extracted_slice_1[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_3 = tensor.extract_slice %extracted_slice_2[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_4 = tensor.extract_slice %arg2[0, 0] [1, 40] [1, 1] : tensor<1x40xi64> to tensor<1x40xi64>
    %17 = tosa.reshape %extracted_slice_4 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi64>) -> tensor<1x1x40xi64>
    %18 = tosa.reshape %17 {new_shape = array<i64: 1, 1, 1, 40>} : (tensor<1x1x40xi64>) -> tensor<1x1x1x40xi64>
    %extracted_slice_5 = tensor.extract_slice %18[0, 0, 0, 0] [1, 1, 1, 40] [1, 1, 1, 1] : tensor<1x1x1x40xi64> to tensor<1x1x1x40xi64>
    %19 = tosa.cast %extracted_slice_5 : (tensor<1x1x1x40xi64>) -> tensor<1x1x1x40xf32>
    %20 = tosa.add %extracted_slice_3, %19 : (tensor<1x1x40x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x1x40x40xf32>
    %cst_6 = arith.constant 0.000000e+00 : f32
    %splat = tensor.splat %cst_6 : tensor<1x1x40x40xf32>
    %21 = arith.cmpf oeq, %20, %splat : tensor<1x1x40x40xf32>
    %extracted_slice_7 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_8 = tensor.extract_slice %extracted_slice_7[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_9 = tensor.extract_slice %extracted_slice_8[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_10 = arith.constant -3.40282347E+38 : f32
    %22 = tensor.empty() : tensor<1x1x40x40xf32>
    %splat_11 = tensor.splat %cst_10 : tensor<1x1x40x40xf32>
    %23 = linalg.generic {indexing_maps = [#map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %splat_11, %extracted_slice_9 : tensor<1x1x40x40xi1>, tensor<1x1x40x40xf32>, tensor<1x1x40x40xf32>) outs(%22 : tensor<1x1x40x40xf32>) {
    ^bb0(%in: i1, %in_843: f32, %in_844: f32, %out: f32):
      %3964 = arith.select %in, %in_843, %in_844 : f32
      linalg.yield %3964 : f32
    } -> tensor<1x1x40x40xf32>
    %extracted_slice_12 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_13 = tensor.extract_slice %extracted_slice_12[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_14 = tensor.extract_slice %extracted_slice_13[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %24 = tensor.empty() : tensor<1x1x40x40xf32>
    %25 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<1x1x40x40xf32>) outs(%extracted_slice_14 : tensor<1x1x40x40xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x40x40xf32>
    %extracted_slice_15 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_16 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_17 = tensor.extract_slice %extracted_slice_16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %inserted_slice = tensor.insert_slice %25 into %extracted_slice_16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> into tensor<1x1x40x40xf32>
    %extracted_slice_18 = tensor.extract_slice %extracted_slice_15[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %inserted_slice_19 = tensor.insert_slice %inserted_slice into %extracted_slice_15[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> into tensor<1x1x40x40xf32>
    %extracted_slice_20 = tensor.extract_slice %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %inserted_slice_21 = tensor.insert_slice %inserted_slice_19 into %16[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> into tensor<1x1x40x40xf32>

    // Generate Rotary Position Embedding (RoPE)
    // %arg3 contains the rotary embedding frequencies (64 dimensions)
    %26 = tosa.reshape %arg3 {new_shape = array<i64: 1, 64>} : (tensor<64xf32>) -> tensor<1x64xf32>
    %extracted_slice_22 = tensor.extract_slice %26[0, 0] [1, 64] [1, 1] : tensor<1x64xf32> to tensor<1x64xf32>
    %27 = tosa.reshape %extracted_slice_22 {new_shape = array<i64: 1, 64, 1>} : (tensor<1x64xf32>) -> tensor<1x64x1xf32>
    %28 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
    %29 = tosa.add %27, %28 : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %extracted_slice_23 = tensor.extract_slice %5[0, 0] [1, 40] [1, 1] : tensor<1x40xi64> to tensor<1x40xi64>
    %30 = tosa.reshape %extracted_slice_23 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x40xi64>) -> tensor<1x1x40xi64>
    %extracted_slice_24 = tensor.extract_slice %30[0, 0, 0] [1, 1, 40] [1, 1, 1] : tensor<1x1x40xi64> to tensor<1x1x40xi64>
    %31 = tosa.cast %extracted_slice_24 : (tensor<1x1x40xi64>) -> tensor<1x1x40xf32>
    %32 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x64x1xf32>}> : () -> tensor<1x64x1xf32>
    %33 = tosa.add %29, %32 : (tensor<1x64x1xf32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %34 = tosa.reshape %33 {new_shape = array<i64: 1, 64, 1>} : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %35 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x40xf32>}> : () -> tensor<1x1x40xf32>
    %36 = tosa.add %31, %35 : (tensor<1x1x40xf32>, tensor<1x1x40xf32>) -> tensor<1x1x40xf32>
    %37 = tosa.reshape %36 {new_shape = array<i64: 1, 1, 40>} : (tensor<1x1x40xf32>) -> tensor<1x1x40xf32>
    %38 = tosa.matmul %34, %37 : (tensor<1x64x1xf32>, tensor<1x1x40xf32>) -> tensor<1x64x40xf32>
    %39 = tosa.reshape %38 {new_shape = array<i64: 1, 64, 40>} : (tensor<1x64x40xf32>) -> tensor<1x64x40xf32>
    %40 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %41 = tosa.transpose %39, %40 : (tensor<1x64x40xf32>, tensor<3xi32>) -> tensor<1x40x64xf32>
    %42 = tosa.reshape %41 {new_shape = array<i64: 1, 40, 1, 64>} : (tensor<1x40x64xf32>) -> tensor<1x40x1x64xf32>
    %43 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x40x2x64xf32>}> : () -> tensor<1x40x2x64xf32>
    %44 = tosa.add %42, %43 : (tensor<1x40x1x64xf32>, tensor<1x40x2x64xf32>) -> tensor<1x40x2x64xf32>
    %45 = tosa.identity %44 : (tensor<1x40x2x64xf32>) -> tensor<1x40x2x64xf32>
    %46 = tosa.reshape %45 {new_shape = array<i64: 1, 40, 128>} : (tensor<1x40x2x64xf32>) -> tensor<1x40x128xf32>
    %47 = tosa.identity %46 : (tensor<1x40x128xf32>) -> tensor<1x40x128xf32>
    %48 = math.cos %47 : tensor<1x40x128xf32>
    %49 = math.sin %47 : tensor<1x40x128xf32>
    %cst_25 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %50 = tosa.reshape %cst_25 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %51 = tosa.mul %48, %50 : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>
    %cst_26 = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %52 = tosa.reshape %cst_26 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %53 = tosa.mul %49, %52 : (tensor<1x40x128xf32>, tensor<1x1x1xf32>) -> tensor<1x40x128xf32>

    // ===========================================
    // 3. LAYER NORMALIZATION (Pre-Attention)
    // ===========================================
    // Apply layer normalization before attention mechanism
    // Weights: %arg4 (1536xf32) - layer norm scale
    // Bias: %arg6 (1536xf32) - layer norm bias

    %54 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32 = arith.constant 2 : i32
    %55 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<1x40x1536xf32>) outs(%54 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %56 = tosa.reduce_sum %55 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %57 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %58 = tosa.reciprocal %57 : (tensor<1xf32>) -> tensor<1xf32>
    %59 = tosa.reshape %58 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %60 = tosa.mul %59, %56 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %61 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %62 = tosa.add %60, %61 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %63 = tosa.rsqrt %62 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %64 = tosa.mul %3, %63 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %65 = tosa.reshape %arg4 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %66 = tosa.mul %65, %64 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>

    // ===========================================
    // 4. MULTI-HEAD ATTENTION MECHANISM
    // ===========================================
    // Generate Query, Key, Value projections for attention
    // Model has 12 attention heads, 128 dimensions per head

    %67 = tosa.reshape %66 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %68 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %69 = tosa.transpose %arg5, %68 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %70 = tosa.reshape %67 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %71 = tosa.reshape %69 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %72 = tosa.matmul %70, %71 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %73 = tosa.reshape %72 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %74 = tosa.reshape %arg6 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %75 = tosa.add %74, %73 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %76 = tosa.reshape %75 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>

    // Query Projection: %arg7 (256x1536) -> Q weights
    %77 = tosa.reshape %66 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %78 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %79 = tosa.transpose %arg7, %78 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %80 = tosa.reshape %77 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %81 = tosa.reshape %79 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %82 = tosa.matmul %80, %81 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %83 = tosa.reshape %82 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %84 = tosa.reshape %arg8 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %85 = tosa.add %84, %83 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %86 = tosa.reshape %85 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>

    // Key Projection: %arg9 (256x1536) -> K weights
    %87 = tosa.reshape %66 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %88 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %89 = tosa.transpose %arg9, %88 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %90 = tosa.reshape %87 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %91 = tosa.reshape %89 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %92 = tosa.matmul %90, %91 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %93 = tosa.reshape %92 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %94 = tosa.reshape %arg10 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %95 = tosa.add %94, %93 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %96 = tosa.reshape %95 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    // Apply RoPE (Rotary Position Embedding) to Q, K vectors
    // Reshape for multi-head attention: 12 heads x 128 dims per head

    %97 = tosa.reshape %76 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %98 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %99 = tosa.transpose %97, %98 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %100 = tosa.reshape %86 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %101 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %102 = tosa.transpose %100, %101 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %103 = tosa.reshape %96 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %104 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %105 = tosa.transpose %103, %104 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %106 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %107 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>

    // Apply RoPE to Query (Q) vectors
    %108 = tosa.mul %99, %106 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_27 = tensor.extract_slice %99[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_28 = tensor.extract_slice %99[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %109 = tensor.empty() : tensor<1x12x40x64xf32>
    %110 = linalg.negf ins(%extracted_slice_28 : tensor<1x12x40x64xf32>) outs(%109 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %111 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_29 = tensor.insert_slice %110 into %111[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_30 = tensor.insert_slice %extracted_slice_27 into %inserted_slice_29[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %112 = tosa.mul %inserted_slice_30, %107 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %113 = tosa.add %108, %112 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>

    // Apply RoPE to Key (K) vectors
    %114 = tosa.mul %102, %106 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_31 = tensor.extract_slice %102[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_32 = tensor.extract_slice %102[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %115 = tensor.empty() : tensor<1x2x40x64xf32>
    %116 = linalg.negf ins(%extracted_slice_32 : tensor<1x2x40x64xf32>) outs(%115 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %117 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_33 = tensor.insert_slice %116 into %117[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_34 = tensor.insert_slice %extracted_slice_31 into %inserted_slice_33[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %118 = tosa.mul %inserted_slice_34, %107 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %119 = tosa.add %114, %118 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    // Reshape K vectors for attention computation
    %extracted_slice_35 = tensor.extract_slice %119[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_36 = tensor.extract_slice %extracted_slice_35[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %120 = tosa.reshape %extracted_slice_36 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_37 = tensor.extract_slice %120[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_38 = tensor.extract_slice %extracted_slice_37[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %121 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %122 = tosa.add %extracted_slice_38, %121 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %123 = tosa.identity %122 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %124 = tosa.reshape %123 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>

    // Reshape V vectors for attention computation
    %extracted_slice_39 = tensor.extract_slice %105[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_40 = tensor.extract_slice %extracted_slice_39[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %125 = tosa.reshape %extracted_slice_40 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_41 = tensor.extract_slice %125[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_42 = tensor.extract_slice %extracted_slice_41[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %126 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %127 = tosa.add %extracted_slice_42, %126 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %128 = tosa.identity %127 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %129 = tosa.reshape %128 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>

    // ===========================================
    // 5. ATTENTION SCORE COMPUTATION
    // ===========================================
    // Compute attention scores: Q @ K^T / sqrt(d_k)
    // Apply causal mask and softmax normalization

    %extracted_slice_43 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_44 = tensor.extract_slice %extracted_slice_43[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_45 = tensor.extract_slice %extracted_slice_44[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_46 = arith.constant 0.000000e+00 : f32
    %splat_47 = tensor.splat %cst_46 : tensor<40x40xf32>
    %130 = tosa.reshape %extracted_slice_45 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %131 = tosa.add %splat_47, %130 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %132 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %133 = tosa.transpose %124, %132 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %134 = tosa.reshape %113 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %135 = tosa.reshape %133 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %136 = tosa.matmul %134, %135 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_48 = arith.constant 0.0883883461 : f32
    %splat_49 = tensor.splat %cst_48 : tensor<12x40x40xf32>
    %137 = tosa.mul %136, %splat_49 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %138 = tosa.reshape %131 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %139 = tosa.add %137, %138 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>

    // Apply softmax with numerical stability (log-sum-exp trick)
    %140 = tosa.reduce_max %139 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %141 = tosa.sub %139, %140 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %142 = math.exp %141 : tensor<12x40x40xf32>
    %143 = tosa.reduce_sum %142 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %144 = tosa.log %143 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %145 = tosa.add %140, %144 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %146 = tosa.sub %139, %145 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %147 = math.exp %146 : tensor<12x40x40xf32>
    %148 = tosa.reshape %145 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %149 = tosa.reshape %129 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %150 = tosa.matmul %147, %149 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %151 = tosa.reshape %150 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %152 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %153 = tosa.transpose %151, %152 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %154 = tosa.reshape %153 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>

    // ===========================================
    // 6. OUTPUT PROJECTION & RESIDUAL CONNECTION
    // ===========================================
    // Project attention output back to hidden dimension
    // Add residual connection (attention output + input)

    %155 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %156 = tosa.transpose %arg11, %155 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %157 = tosa.reshape %154 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_50 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %158 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%157, %156 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_50 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %159 = tosa.reshape %158 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %160 = tosa.add %3, %159 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>

    // ===========================================
    // 7. LAYER NORMALIZATION (Pre-FFN)
    // ===========================================
    // Apply layer normalization before feed-forward network

    %161 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_51 = arith.constant 2 : i32
    %162 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%160 : tensor<1x40x1536xf32>) outs(%161 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_51 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %163 = tosa.reduce_sum %162 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %164 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %165 = tosa.reciprocal %164 : (tensor<1xf32>) -> tensor<1xf32>
    %166 = tosa.reshape %165 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %167 = tosa.mul %166, %163 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %168 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %169 = tosa.add %167, %168 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %170 = tosa.rsqrt %169 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %171 = tosa.mul %160, %170 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %172 = tosa.reshape %arg12 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %173 = tosa.mul %172, %171 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>

    // ===========================================
    // 8. FEED-FORWARD NETWORK (FFN)
    // ===========================================
    // DeepSeek uses SwiGLU activation: x * sigmoid(x) * gate
    // FFN expansion ratio: 1536 -> 8960 -> 1536

    %174 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %175 = tosa.transpose %arg13, %174 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %176 = tosa.reshape %173 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_52 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %177 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%176, %175 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_52 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %178 = tosa.reshape %177 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %179 = tosa.sigmoid %178 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %180 = tosa.mul %178, %179 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %181 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %182 = tosa.transpose %arg14, %181 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %183 = tosa.reshape %173 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_53 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %184 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%183, %182 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_53 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %185 = tosa.reshape %184 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %186 = tosa.mul %180, %185 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %187 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %188 = tosa.transpose %arg15, %187 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %189 = tosa.reshape %186 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_54 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %190 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%189, %188 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_54 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %191 = tosa.reshape %190 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %192 = tosa.add %160, %191 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %193 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_55 = arith.constant 2 : i32
    %194 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192 : tensor<1x40x1536xf32>) outs(%193 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_55 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %195 = tosa.reduce_sum %194 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %196 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %197 = tosa.reciprocal %196 : (tensor<1xf32>) -> tensor<1xf32>
    %198 = tosa.reshape %197 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %199 = tosa.mul %198, %195 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %200 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %201 = tosa.add %199, %200 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %202 = tosa.rsqrt %201 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %203 = tosa.mul %192, %202 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %204 = tosa.reshape %arg16 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %205 = tosa.mul %204, %203 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %206 = tosa.reshape %205 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %207 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %208 = tosa.transpose %arg17, %207 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %209 = tosa.reshape %206 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %210 = tosa.reshape %208 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %211 = tosa.matmul %209, %210 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>

    // ===========================================
    // 9. REMAINING TRANSFORMER LAYERS (Layer 2-28)
    // ===========================================
    // The following operations represent the remaining 27 transformer layers
    // Each layer follows the same pattern as Layer 1:
    // 1. Layer Normalization (Pre-Attention)
    // 2. Multi-Head Self-Attention with RoPE
    // 3. Residual Connection + Output Projection
    // 4. Layer Normalization (Pre-FFN)
    // 5. SwiGLU Feed-Forward Network
    // 6. Residual Connection
    //
    // Total Architecture: 28 Transformer Layers
    // - Each layer processes the same sequence length (40 tokens)
    // - Same hidden dimension (1536) and attention heads (12)
    // - Same FFN expansion ratio (1536 -> 8960 -> 1536)
    // - RoPE applied to Q, K vectors in each attention layer
    // - SwiGLU activation in each FFN layer
    //
    // Note: The parameters (arg17-arg340) represent the weights and biases
    // for all 28 transformer layers, with each layer having its own
    // set of learned parameters for attention and FFN operations.
    // ===========================================
    %212 = tosa.reshape %211 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %213 = tosa.reshape %arg18 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %214 = tosa.add %213, %212 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %215 = tosa.reshape %214 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %216 = tosa.reshape %205 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %217 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %218 = tosa.transpose %arg19, %217 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %219 = tosa.reshape %216 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %220 = tosa.reshape %218 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %221 = tosa.matmul %219, %220 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %222 = tosa.reshape %221 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %223 = tosa.reshape %arg20 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %224 = tosa.add %223, %222 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %225 = tosa.reshape %224 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %226 = tosa.reshape %205 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %227 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %228 = tosa.transpose %arg21, %227 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %229 = tosa.reshape %226 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %230 = tosa.reshape %228 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %231 = tosa.matmul %229, %230 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %232 = tosa.reshape %231 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %233 = tosa.reshape %arg22 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %234 = tosa.add %233, %232 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %235 = tosa.reshape %234 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %236 = tosa.reshape %215 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %237 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %238 = tosa.transpose %236, %237 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %239 = tosa.reshape %225 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %240 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %241 = tosa.transpose %239, %240 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %242 = tosa.reshape %235 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %243 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %244 = tosa.transpose %242, %243 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %245 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %246 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %247 = tosa.mul %238, %245 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_56 = tensor.extract_slice %238[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_57 = tensor.extract_slice %238[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %248 = tensor.empty() : tensor<1x12x40x64xf32>
    %249 = linalg.negf ins(%extracted_slice_57 : tensor<1x12x40x64xf32>) outs(%248 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %250 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_58 = tensor.insert_slice %249 into %250[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_59 = tensor.insert_slice %extracted_slice_56 into %inserted_slice_58[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %251 = tosa.mul %inserted_slice_59, %246 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %252 = tosa.add %247, %251 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %253 = tosa.mul %241, %245 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_60 = tensor.extract_slice %241[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_61 = tensor.extract_slice %241[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %254 = tensor.empty() : tensor<1x2x40x64xf32>
    %255 = linalg.negf ins(%extracted_slice_61 : tensor<1x2x40x64xf32>) outs(%254 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %256 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_62 = tensor.insert_slice %255 into %256[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_63 = tensor.insert_slice %extracted_slice_60 into %inserted_slice_62[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %257 = tosa.mul %inserted_slice_63, %246 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %258 = tosa.add %253, %257 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_64 = tensor.extract_slice %258[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_65 = tensor.extract_slice %extracted_slice_64[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %259 = tosa.reshape %extracted_slice_65 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_66 = tensor.extract_slice %259[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_67 = tensor.extract_slice %extracted_slice_66[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %260 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %261 = tosa.add %extracted_slice_67, %260 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %262 = tosa.identity %261 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %263 = tosa.reshape %262 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_68 = tensor.extract_slice %244[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_69 = tensor.extract_slice %extracted_slice_68[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %264 = tosa.reshape %extracted_slice_69 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_70 = tensor.extract_slice %264[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_71 = tensor.extract_slice %extracted_slice_70[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %265 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %266 = tosa.add %extracted_slice_71, %265 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %267 = tosa.identity %266 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %268 = tosa.reshape %267 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_72 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_73 = tensor.extract_slice %extracted_slice_72[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_74 = tensor.extract_slice %extracted_slice_73[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_75 = arith.constant 0.000000e+00 : f32
    %splat_76 = tensor.splat %cst_75 : tensor<40x40xf32>
    %269 = tosa.reshape %extracted_slice_74 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %270 = tosa.add %splat_76, %269 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %271 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %272 = tosa.transpose %263, %271 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %273 = tosa.reshape %252 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %274 = tosa.reshape %272 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %275 = tosa.matmul %273, %274 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_77 = arith.constant 0.0883883461 : f32
    %splat_78 = tensor.splat %cst_77 : tensor<12x40x40xf32>
    %276 = tosa.mul %275, %splat_78 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %277 = tosa.reshape %270 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %278 = tosa.add %276, %277 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %279 = tosa.reduce_max %278 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %280 = tosa.sub %278, %279 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %281 = math.exp %280 : tensor<12x40x40xf32>
    %282 = tosa.reduce_sum %281 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %283 = tosa.log %282 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %284 = tosa.add %279, %283 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %285 = tosa.sub %278, %284 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %286 = math.exp %285 : tensor<12x40x40xf32>
    %287 = tosa.reshape %284 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %288 = tosa.reshape %268 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %289 = tosa.matmul %286, %288 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %290 = tosa.reshape %289 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %291 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %292 = tosa.transpose %290, %291 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %293 = tosa.reshape %292 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %294 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %295 = tosa.transpose %arg23, %294 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %296 = tosa.reshape %293 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_79 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %297 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%296, %295 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_79 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %298 = tosa.reshape %297 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %299 = tosa.add %192, %298 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %300 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_80 = arith.constant 2 : i32
    %301 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%299 : tensor<1x40x1536xf32>) outs(%300 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_80 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %302 = tosa.reduce_sum %301 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %303 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %304 = tosa.reciprocal %303 : (tensor<1xf32>) -> tensor<1xf32>
    %305 = tosa.reshape %304 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %306 = tosa.mul %305, %302 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %307 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %308 = tosa.add %306, %307 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %309 = tosa.rsqrt %308 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %310 = tosa.mul %299, %309 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %311 = tosa.reshape %arg24 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %312 = tosa.mul %311, %310 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %313 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %314 = tosa.transpose %arg25, %313 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %315 = tosa.reshape %312 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_81 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %316 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%315, %314 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_81 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %317 = tosa.reshape %316 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %318 = tosa.sigmoid %317 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %319 = tosa.mul %317, %318 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %320 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %321 = tosa.transpose %arg26, %320 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %322 = tosa.reshape %312 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_82 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %323 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%322, %321 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_82 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %324 = tosa.reshape %323 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %325 = tosa.mul %319, %324 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %326 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %327 = tosa.transpose %arg27, %326 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %328 = tosa.reshape %325 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_83 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %329 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%328, %327 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_83 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %330 = tosa.reshape %329 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %331 = tosa.add %299, %330 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %332 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_84 = arith.constant 2 : i32
    %333 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%331 : tensor<1x40x1536xf32>) outs(%332 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_84 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %334 = tosa.reduce_sum %333 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %335 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %336 = tosa.reciprocal %335 : (tensor<1xf32>) -> tensor<1xf32>
    %337 = tosa.reshape %336 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %338 = tosa.mul %337, %334 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %339 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %340 = tosa.add %338, %339 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %341 = tosa.rsqrt %340 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %342 = tosa.mul %331, %341 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %343 = tosa.reshape %arg28 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %344 = tosa.mul %343, %342 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %345 = tosa.reshape %344 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %346 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %347 = tosa.transpose %arg29, %346 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %348 = tosa.reshape %345 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %349 = tosa.reshape %347 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %350 = tosa.matmul %348, %349 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %351 = tosa.reshape %350 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %352 = tosa.reshape %arg30 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %353 = tosa.add %352, %351 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %354 = tosa.reshape %353 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %355 = tosa.reshape %344 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %356 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %357 = tosa.transpose %arg31, %356 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %358 = tosa.reshape %355 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %359 = tosa.reshape %357 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %360 = tosa.matmul %358, %359 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %361 = tosa.reshape %360 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %362 = tosa.reshape %arg32 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %363 = tosa.add %362, %361 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %364 = tosa.reshape %363 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %365 = tosa.reshape %344 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %366 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %367 = tosa.transpose %arg33, %366 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %368 = tosa.reshape %365 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %369 = tosa.reshape %367 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %370 = tosa.matmul %368, %369 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %371 = tosa.reshape %370 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %372 = tosa.reshape %arg34 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %373 = tosa.add %372, %371 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %374 = tosa.reshape %373 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %375 = tosa.reshape %354 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %376 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %377 = tosa.transpose %375, %376 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %378 = tosa.reshape %364 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %379 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %380 = tosa.transpose %378, %379 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %381 = tosa.reshape %374 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %382 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %383 = tosa.transpose %381, %382 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %384 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %385 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %386 = tosa.mul %377, %384 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_85 = tensor.extract_slice %377[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_86 = tensor.extract_slice %377[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %387 = tensor.empty() : tensor<1x12x40x64xf32>
    %388 = linalg.negf ins(%extracted_slice_86 : tensor<1x12x40x64xf32>) outs(%387 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %389 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_87 = tensor.insert_slice %388 into %389[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_88 = tensor.insert_slice %extracted_slice_85 into %inserted_slice_87[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %390 = tosa.mul %inserted_slice_88, %385 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %391 = tosa.add %386, %390 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %392 = tosa.mul %380, %384 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_89 = tensor.extract_slice %380[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_90 = tensor.extract_slice %380[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %393 = tensor.empty() : tensor<1x2x40x64xf32>
    %394 = linalg.negf ins(%extracted_slice_90 : tensor<1x2x40x64xf32>) outs(%393 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %395 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_91 = tensor.insert_slice %394 into %395[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_92 = tensor.insert_slice %extracted_slice_89 into %inserted_slice_91[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %396 = tosa.mul %inserted_slice_92, %385 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %397 = tosa.add %392, %396 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_93 = tensor.extract_slice %397[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_94 = tensor.extract_slice %extracted_slice_93[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %398 = tosa.reshape %extracted_slice_94 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_95 = tensor.extract_slice %398[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_96 = tensor.extract_slice %extracted_slice_95[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %399 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %400 = tosa.add %extracted_slice_96, %399 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %401 = tosa.identity %400 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %402 = tosa.reshape %401 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_97 = tensor.extract_slice %383[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_98 = tensor.extract_slice %extracted_slice_97[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %403 = tosa.reshape %extracted_slice_98 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_99 = tensor.extract_slice %403[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_100 = tensor.extract_slice %extracted_slice_99[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %404 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %405 = tosa.add %extracted_slice_100, %404 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %406 = tosa.identity %405 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %407 = tosa.reshape %406 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_101 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_102 = tensor.extract_slice %extracted_slice_101[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_103 = tensor.extract_slice %extracted_slice_102[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_104 = arith.constant 0.000000e+00 : f32
    %splat_105 = tensor.splat %cst_104 : tensor<40x40xf32>
    %408 = tosa.reshape %extracted_slice_103 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %409 = tosa.add %splat_105, %408 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %410 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %411 = tosa.transpose %402, %410 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %412 = tosa.reshape %391 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %413 = tosa.reshape %411 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %414 = tosa.matmul %412, %413 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_106 = arith.constant 0.0883883461 : f32
    %splat_107 = tensor.splat %cst_106 : tensor<12x40x40xf32>
    %415 = tosa.mul %414, %splat_107 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %416 = tosa.reshape %409 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %417 = tosa.add %415, %416 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %418 = tosa.reduce_max %417 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %419 = tosa.sub %417, %418 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %420 = math.exp %419 : tensor<12x40x40xf32>
    %421 = tosa.reduce_sum %420 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %422 = tosa.log %421 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %423 = tosa.add %418, %422 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %424 = tosa.sub %417, %423 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %425 = math.exp %424 : tensor<12x40x40xf32>
    %426 = tosa.reshape %423 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %427 = tosa.reshape %407 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %428 = tosa.matmul %425, %427 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %429 = tosa.reshape %428 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %430 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %431 = tosa.transpose %429, %430 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %432 = tosa.reshape %431 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %433 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %434 = tosa.transpose %arg35, %433 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %435 = tosa.reshape %432 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_108 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %436 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%435, %434 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_108 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %437 = tosa.reshape %436 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %438 = tosa.add %331, %437 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %439 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_109 = arith.constant 2 : i32
    %440 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%438 : tensor<1x40x1536xf32>) outs(%439 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_109 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %441 = tosa.reduce_sum %440 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %442 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %443 = tosa.reciprocal %442 : (tensor<1xf32>) -> tensor<1xf32>
    %444 = tosa.reshape %443 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %445 = tosa.mul %444, %441 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %446 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %447 = tosa.add %445, %446 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %448 = tosa.rsqrt %447 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %449 = tosa.mul %438, %448 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %450 = tosa.reshape %arg36 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %451 = tosa.mul %450, %449 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %452 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %453 = tosa.transpose %arg37, %452 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %454 = tosa.reshape %451 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_110 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %455 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%454, %453 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_110 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %456 = tosa.reshape %455 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %457 = tosa.sigmoid %456 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %458 = tosa.mul %456, %457 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %459 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %460 = tosa.transpose %arg38, %459 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %461 = tosa.reshape %451 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_111 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %462 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%461, %460 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_111 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %463 = tosa.reshape %462 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %464 = tosa.mul %458, %463 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %465 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %466 = tosa.transpose %arg39, %465 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %467 = tosa.reshape %464 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_112 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %468 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%467, %466 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_112 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %469 = tosa.reshape %468 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %470 = tosa.add %438, %469 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %471 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_113 = arith.constant 2 : i32
    %472 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%470 : tensor<1x40x1536xf32>) outs(%471 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_113 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %473 = tosa.reduce_sum %472 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %474 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %475 = tosa.reciprocal %474 : (tensor<1xf32>) -> tensor<1xf32>
    %476 = tosa.reshape %475 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %477 = tosa.mul %476, %473 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %478 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %479 = tosa.add %477, %478 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %480 = tosa.rsqrt %479 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %481 = tosa.mul %470, %480 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %482 = tosa.reshape %arg40 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %483 = tosa.mul %482, %481 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %484 = tosa.reshape %483 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %485 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %486 = tosa.transpose %arg41, %485 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %487 = tosa.reshape %484 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %488 = tosa.reshape %486 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %489 = tosa.matmul %487, %488 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %490 = tosa.reshape %489 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %491 = tosa.reshape %arg42 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %492 = tosa.add %491, %490 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %493 = tosa.reshape %492 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %494 = tosa.reshape %483 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %495 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %496 = tosa.transpose %arg43, %495 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %497 = tosa.reshape %494 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %498 = tosa.reshape %496 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %499 = tosa.matmul %497, %498 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %500 = tosa.reshape %499 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %501 = tosa.reshape %arg44 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %502 = tosa.add %501, %500 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %503 = tosa.reshape %502 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %504 = tosa.reshape %483 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %505 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %506 = tosa.transpose %arg45, %505 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %507 = tosa.reshape %504 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %508 = tosa.reshape %506 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %509 = tosa.matmul %507, %508 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %510 = tosa.reshape %509 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %511 = tosa.reshape %arg46 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %512 = tosa.add %511, %510 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %513 = tosa.reshape %512 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %514 = tosa.reshape %493 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %515 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %516 = tosa.transpose %514, %515 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %517 = tosa.reshape %503 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %518 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %519 = tosa.transpose %517, %518 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %520 = tosa.reshape %513 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %521 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %522 = tosa.transpose %520, %521 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %523 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %524 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %525 = tosa.mul %516, %523 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_114 = tensor.extract_slice %516[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_115 = tensor.extract_slice %516[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %526 = tensor.empty() : tensor<1x12x40x64xf32>
    %527 = linalg.negf ins(%extracted_slice_115 : tensor<1x12x40x64xf32>) outs(%526 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %528 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_116 = tensor.insert_slice %527 into %528[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_117 = tensor.insert_slice %extracted_slice_114 into %inserted_slice_116[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %529 = tosa.mul %inserted_slice_117, %524 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %530 = tosa.add %525, %529 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %531 = tosa.mul %519, %523 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_118 = tensor.extract_slice %519[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_119 = tensor.extract_slice %519[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %532 = tensor.empty() : tensor<1x2x40x64xf32>
    %533 = linalg.negf ins(%extracted_slice_119 : tensor<1x2x40x64xf32>) outs(%532 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %534 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_120 = tensor.insert_slice %533 into %534[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_121 = tensor.insert_slice %extracted_slice_118 into %inserted_slice_120[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %535 = tosa.mul %inserted_slice_121, %524 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %536 = tosa.add %531, %535 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_122 = tensor.extract_slice %536[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_123 = tensor.extract_slice %extracted_slice_122[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %537 = tosa.reshape %extracted_slice_123 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_124 = tensor.extract_slice %537[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_125 = tensor.extract_slice %extracted_slice_124[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %538 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %539 = tosa.add %extracted_slice_125, %538 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %540 = tosa.identity %539 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %541 = tosa.reshape %540 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_126 = tensor.extract_slice %522[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_127 = tensor.extract_slice %extracted_slice_126[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %542 = tosa.reshape %extracted_slice_127 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_128 = tensor.extract_slice %542[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_129 = tensor.extract_slice %extracted_slice_128[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %543 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %544 = tosa.add %extracted_slice_129, %543 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %545 = tosa.identity %544 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %546 = tosa.reshape %545 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_130 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_131 = tensor.extract_slice %extracted_slice_130[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_132 = tensor.extract_slice %extracted_slice_131[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_133 = arith.constant 0.000000e+00 : f32
    %splat_134 = tensor.splat %cst_133 : tensor<40x40xf32>
    %547 = tosa.reshape %extracted_slice_132 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %548 = tosa.add %splat_134, %547 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %549 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %550 = tosa.transpose %541, %549 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %551 = tosa.reshape %530 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %552 = tosa.reshape %550 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %553 = tosa.matmul %551, %552 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_135 = arith.constant 0.0883883461 : f32
    %splat_136 = tensor.splat %cst_135 : tensor<12x40x40xf32>
    %554 = tosa.mul %553, %splat_136 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %555 = tosa.reshape %548 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %556 = tosa.add %554, %555 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %557 = tosa.reduce_max %556 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %558 = tosa.sub %556, %557 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %559 = math.exp %558 : tensor<12x40x40xf32>
    %560 = tosa.reduce_sum %559 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %561 = tosa.log %560 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %562 = tosa.add %557, %561 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %563 = tosa.sub %556, %562 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %564 = math.exp %563 : tensor<12x40x40xf32>
    %565 = tosa.reshape %562 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %566 = tosa.reshape %546 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %567 = tosa.matmul %564, %566 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %568 = tosa.reshape %567 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %569 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %570 = tosa.transpose %568, %569 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %571 = tosa.reshape %570 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %572 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %573 = tosa.transpose %arg47, %572 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %574 = tosa.reshape %571 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_137 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %575 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%574, %573 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_137 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %576 = tosa.reshape %575 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %577 = tosa.add %470, %576 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %578 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_138 = arith.constant 2 : i32
    %579 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%577 : tensor<1x40x1536xf32>) outs(%578 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_138 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %580 = tosa.reduce_sum %579 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %581 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %582 = tosa.reciprocal %581 : (tensor<1xf32>) -> tensor<1xf32>
    %583 = tosa.reshape %582 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %584 = tosa.mul %583, %580 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %585 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %586 = tosa.add %584, %585 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %587 = tosa.rsqrt %586 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %588 = tosa.mul %577, %587 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %589 = tosa.reshape %arg48 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %590 = tosa.mul %589, %588 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %591 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %592 = tosa.transpose %arg49, %591 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %593 = tosa.reshape %590 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_139 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %594 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%593, %592 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_139 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %595 = tosa.reshape %594 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %596 = tosa.sigmoid %595 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %597 = tosa.mul %595, %596 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %598 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %599 = tosa.transpose %arg50, %598 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %600 = tosa.reshape %590 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_140 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %601 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%600, %599 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_140 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %602 = tosa.reshape %601 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %603 = tosa.mul %597, %602 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %604 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %605 = tosa.transpose %arg51, %604 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %606 = tosa.reshape %603 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_141 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %607 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%606, %605 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_141 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %608 = tosa.reshape %607 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %609 = tosa.add %577, %608 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %610 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_142 = arith.constant 2 : i32
    %611 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%609 : tensor<1x40x1536xf32>) outs(%610 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_142 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %612 = tosa.reduce_sum %611 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %613 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %614 = tosa.reciprocal %613 : (tensor<1xf32>) -> tensor<1xf32>
    %615 = tosa.reshape %614 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %616 = tosa.mul %615, %612 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %617 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %618 = tosa.add %616, %617 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %619 = tosa.rsqrt %618 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %620 = tosa.mul %609, %619 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %621 = tosa.reshape %arg52 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %622 = tosa.mul %621, %620 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %623 = tosa.reshape %622 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %624 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %625 = tosa.transpose %arg53, %624 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %626 = tosa.reshape %623 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %627 = tosa.reshape %625 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %628 = tosa.matmul %626, %627 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %629 = tosa.reshape %628 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %630 = tosa.reshape %arg54 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %631 = tosa.add %630, %629 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %632 = tosa.reshape %631 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %633 = tosa.reshape %622 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %634 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %635 = tosa.transpose %arg55, %634 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %636 = tosa.reshape %633 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %637 = tosa.reshape %635 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %638 = tosa.matmul %636, %637 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %639 = tosa.reshape %638 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %640 = tosa.reshape %arg56 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %641 = tosa.add %640, %639 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %642 = tosa.reshape %641 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %643 = tosa.reshape %622 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %644 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %645 = tosa.transpose %arg57, %644 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %646 = tosa.reshape %643 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %647 = tosa.reshape %645 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %648 = tosa.matmul %646, %647 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %649 = tosa.reshape %648 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %650 = tosa.reshape %arg58 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %651 = tosa.add %650, %649 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %652 = tosa.reshape %651 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %653 = tosa.reshape %632 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %654 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %655 = tosa.transpose %653, %654 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %656 = tosa.reshape %642 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %657 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %658 = tosa.transpose %656, %657 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %659 = tosa.reshape %652 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %660 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %661 = tosa.transpose %659, %660 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %662 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %663 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %664 = tosa.mul %655, %662 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_143 = tensor.extract_slice %655[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_144 = tensor.extract_slice %655[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %665 = tensor.empty() : tensor<1x12x40x64xf32>
    %666 = linalg.negf ins(%extracted_slice_144 : tensor<1x12x40x64xf32>) outs(%665 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %667 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_145 = tensor.insert_slice %666 into %667[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_146 = tensor.insert_slice %extracted_slice_143 into %inserted_slice_145[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %668 = tosa.mul %inserted_slice_146, %663 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %669 = tosa.add %664, %668 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %670 = tosa.mul %658, %662 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_147 = tensor.extract_slice %658[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_148 = tensor.extract_slice %658[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %671 = tensor.empty() : tensor<1x2x40x64xf32>
    %672 = linalg.negf ins(%extracted_slice_148 : tensor<1x2x40x64xf32>) outs(%671 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %673 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_149 = tensor.insert_slice %672 into %673[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_150 = tensor.insert_slice %extracted_slice_147 into %inserted_slice_149[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %674 = tosa.mul %inserted_slice_150, %663 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %675 = tosa.add %670, %674 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_151 = tensor.extract_slice %675[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_152 = tensor.extract_slice %extracted_slice_151[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %676 = tosa.reshape %extracted_slice_152 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_153 = tensor.extract_slice %676[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_154 = tensor.extract_slice %extracted_slice_153[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %677 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %678 = tosa.add %extracted_slice_154, %677 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %679 = tosa.identity %678 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %680 = tosa.reshape %679 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_155 = tensor.extract_slice %661[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_156 = tensor.extract_slice %extracted_slice_155[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %681 = tosa.reshape %extracted_slice_156 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_157 = tensor.extract_slice %681[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_158 = tensor.extract_slice %extracted_slice_157[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %682 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %683 = tosa.add %extracted_slice_158, %682 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %684 = tosa.identity %683 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %685 = tosa.reshape %684 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_159 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_160 = tensor.extract_slice %extracted_slice_159[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_161 = tensor.extract_slice %extracted_slice_160[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_162 = arith.constant 0.000000e+00 : f32
    %splat_163 = tensor.splat %cst_162 : tensor<40x40xf32>
    %686 = tosa.reshape %extracted_slice_161 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %687 = tosa.add %splat_163, %686 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %688 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %689 = tosa.transpose %680, %688 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %690 = tosa.reshape %669 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %691 = tosa.reshape %689 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %692 = tosa.matmul %690, %691 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_164 = arith.constant 0.0883883461 : f32
    %splat_165 = tensor.splat %cst_164 : tensor<12x40x40xf32>
    %693 = tosa.mul %692, %splat_165 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %694 = tosa.reshape %687 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %695 = tosa.add %693, %694 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %696 = tosa.reduce_max %695 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %697 = tosa.sub %695, %696 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %698 = math.exp %697 : tensor<12x40x40xf32>
    %699 = tosa.reduce_sum %698 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %700 = tosa.log %699 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %701 = tosa.add %696, %700 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %702 = tosa.sub %695, %701 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %703 = math.exp %702 : tensor<12x40x40xf32>
    %704 = tosa.reshape %701 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %705 = tosa.reshape %685 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %706 = tosa.matmul %703, %705 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %707 = tosa.reshape %706 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %708 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %709 = tosa.transpose %707, %708 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %710 = tosa.reshape %709 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %711 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %712 = tosa.transpose %arg59, %711 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %713 = tosa.reshape %710 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_166 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %714 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%713, %712 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_166 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %715 = tosa.reshape %714 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %716 = tosa.add %609, %715 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %717 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_167 = arith.constant 2 : i32
    %718 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%716 : tensor<1x40x1536xf32>) outs(%717 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_167 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %719 = tosa.reduce_sum %718 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %720 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %721 = tosa.reciprocal %720 : (tensor<1xf32>) -> tensor<1xf32>
    %722 = tosa.reshape %721 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %723 = tosa.mul %722, %719 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %724 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %725 = tosa.add %723, %724 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %726 = tosa.rsqrt %725 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %727 = tosa.mul %716, %726 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %728 = tosa.reshape %arg60 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %729 = tosa.mul %728, %727 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %730 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %731 = tosa.transpose %arg61, %730 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %732 = tosa.reshape %729 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_168 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %733 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%732, %731 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_168 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %734 = tosa.reshape %733 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %735 = tosa.sigmoid %734 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %736 = tosa.mul %734, %735 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %737 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %738 = tosa.transpose %arg62, %737 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %739 = tosa.reshape %729 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_169 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %740 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%739, %738 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_169 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %741 = tosa.reshape %740 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %742 = tosa.mul %736, %741 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %743 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %744 = tosa.transpose %arg63, %743 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %745 = tosa.reshape %742 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_170 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %746 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%745, %744 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_170 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %747 = tosa.reshape %746 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %748 = tosa.add %716, %747 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %749 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_171 = arith.constant 2 : i32
    %750 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%748 : tensor<1x40x1536xf32>) outs(%749 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_171 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %751 = tosa.reduce_sum %750 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %752 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %753 = tosa.reciprocal %752 : (tensor<1xf32>) -> tensor<1xf32>
    %754 = tosa.reshape %753 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %755 = tosa.mul %754, %751 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %756 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %757 = tosa.add %755, %756 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %758 = tosa.rsqrt %757 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %759 = tosa.mul %748, %758 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %760 = tosa.reshape %arg64 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %761 = tosa.mul %760, %759 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %762 = tosa.reshape %761 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %763 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %764 = tosa.transpose %arg65, %763 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %765 = tosa.reshape %762 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %766 = tosa.reshape %764 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %767 = tosa.matmul %765, %766 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %768 = tosa.reshape %767 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %769 = tosa.reshape %arg66 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %770 = tosa.add %769, %768 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %771 = tosa.reshape %770 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %772 = tosa.reshape %761 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %773 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %774 = tosa.transpose %arg67, %773 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %775 = tosa.reshape %772 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %776 = tosa.reshape %774 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %777 = tosa.matmul %775, %776 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %778 = tosa.reshape %777 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %779 = tosa.reshape %arg68 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %780 = tosa.add %779, %778 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %781 = tosa.reshape %780 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %782 = tosa.reshape %761 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %783 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %784 = tosa.transpose %arg69, %783 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %785 = tosa.reshape %782 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %786 = tosa.reshape %784 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %787 = tosa.matmul %785, %786 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %788 = tosa.reshape %787 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %789 = tosa.reshape %arg70 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %790 = tosa.add %789, %788 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %791 = tosa.reshape %790 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %792 = tosa.reshape %771 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %793 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %794 = tosa.transpose %792, %793 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %795 = tosa.reshape %781 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %796 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %797 = tosa.transpose %795, %796 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %798 = tosa.reshape %791 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %799 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %800 = tosa.transpose %798, %799 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %801 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %802 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %803 = tosa.mul %794, %801 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_172 = tensor.extract_slice %794[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_173 = tensor.extract_slice %794[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %804 = tensor.empty() : tensor<1x12x40x64xf32>
    %805 = linalg.negf ins(%extracted_slice_173 : tensor<1x12x40x64xf32>) outs(%804 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %806 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_174 = tensor.insert_slice %805 into %806[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_175 = tensor.insert_slice %extracted_slice_172 into %inserted_slice_174[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %807 = tosa.mul %inserted_slice_175, %802 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %808 = tosa.add %803, %807 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %809 = tosa.mul %797, %801 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_176 = tensor.extract_slice %797[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_177 = tensor.extract_slice %797[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %810 = tensor.empty() : tensor<1x2x40x64xf32>
    %811 = linalg.negf ins(%extracted_slice_177 : tensor<1x2x40x64xf32>) outs(%810 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %812 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_178 = tensor.insert_slice %811 into %812[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_179 = tensor.insert_slice %extracted_slice_176 into %inserted_slice_178[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %813 = tosa.mul %inserted_slice_179, %802 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %814 = tosa.add %809, %813 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_180 = tensor.extract_slice %814[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_181 = tensor.extract_slice %extracted_slice_180[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %815 = tosa.reshape %extracted_slice_181 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_182 = tensor.extract_slice %815[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_183 = tensor.extract_slice %extracted_slice_182[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %816 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %817 = tosa.add %extracted_slice_183, %816 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %818 = tosa.identity %817 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %819 = tosa.reshape %818 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_184 = tensor.extract_slice %800[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_185 = tensor.extract_slice %extracted_slice_184[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %820 = tosa.reshape %extracted_slice_185 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_186 = tensor.extract_slice %820[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_187 = tensor.extract_slice %extracted_slice_186[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %821 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %822 = tosa.add %extracted_slice_187, %821 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %823 = tosa.identity %822 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %824 = tosa.reshape %823 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_188 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_189 = tensor.extract_slice %extracted_slice_188[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_190 = tensor.extract_slice %extracted_slice_189[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_191 = arith.constant 0.000000e+00 : f32
    %splat_192 = tensor.splat %cst_191 : tensor<40x40xf32>
    %825 = tosa.reshape %extracted_slice_190 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %826 = tosa.add %splat_192, %825 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %827 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %828 = tosa.transpose %819, %827 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %829 = tosa.reshape %808 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %830 = tosa.reshape %828 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %831 = tosa.matmul %829, %830 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_193 = arith.constant 0.0883883461 : f32
    %splat_194 = tensor.splat %cst_193 : tensor<12x40x40xf32>
    %832 = tosa.mul %831, %splat_194 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %833 = tosa.reshape %826 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %834 = tosa.add %832, %833 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %835 = tosa.reduce_max %834 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %836 = tosa.sub %834, %835 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %837 = math.exp %836 : tensor<12x40x40xf32>
    %838 = tosa.reduce_sum %837 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %839 = tosa.log %838 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %840 = tosa.add %835, %839 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %841 = tosa.sub %834, %840 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %842 = math.exp %841 : tensor<12x40x40xf32>
    %843 = tosa.reshape %840 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %844 = tosa.reshape %824 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %845 = tosa.matmul %842, %844 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %846 = tosa.reshape %845 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %847 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %848 = tosa.transpose %846, %847 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %849 = tosa.reshape %848 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %850 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %851 = tosa.transpose %arg71, %850 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %852 = tosa.reshape %849 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_195 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %853 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%852, %851 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_195 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %854 = tosa.reshape %853 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %855 = tosa.add %748, %854 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %856 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_196 = arith.constant 2 : i32
    %857 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%855 : tensor<1x40x1536xf32>) outs(%856 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_196 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %858 = tosa.reduce_sum %857 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %859 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %860 = tosa.reciprocal %859 : (tensor<1xf32>) -> tensor<1xf32>
    %861 = tosa.reshape %860 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %862 = tosa.mul %861, %858 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %863 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %864 = tosa.add %862, %863 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %865 = tosa.rsqrt %864 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %866 = tosa.mul %855, %865 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %867 = tosa.reshape %arg72 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %868 = tosa.mul %867, %866 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %869 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %870 = tosa.transpose %arg73, %869 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %871 = tosa.reshape %868 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_197 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %872 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%871, %870 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_197 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %873 = tosa.reshape %872 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %874 = tosa.sigmoid %873 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %875 = tosa.mul %873, %874 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %876 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %877 = tosa.transpose %arg74, %876 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %878 = tosa.reshape %868 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_198 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %879 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%878, %877 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_198 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %880 = tosa.reshape %879 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %881 = tosa.mul %875, %880 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %882 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %883 = tosa.transpose %arg75, %882 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %884 = tosa.reshape %881 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_199 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %885 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%884, %883 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_199 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %886 = tosa.reshape %885 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %887 = tosa.add %855, %886 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %888 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_200 = arith.constant 2 : i32
    %889 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%887 : tensor<1x40x1536xf32>) outs(%888 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_200 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %890 = tosa.reduce_sum %889 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %891 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %892 = tosa.reciprocal %891 : (tensor<1xf32>) -> tensor<1xf32>
    %893 = tosa.reshape %892 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %894 = tosa.mul %893, %890 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %895 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %896 = tosa.add %894, %895 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %897 = tosa.rsqrt %896 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %898 = tosa.mul %887, %897 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %899 = tosa.reshape %arg76 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %900 = tosa.mul %899, %898 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %901 = tosa.reshape %900 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %902 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %903 = tosa.transpose %arg77, %902 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %904 = tosa.reshape %901 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %905 = tosa.reshape %903 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %906 = tosa.matmul %904, %905 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %907 = tosa.reshape %906 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %908 = tosa.reshape %arg78 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %909 = tosa.add %908, %907 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %910 = tosa.reshape %909 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %911 = tosa.reshape %900 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %912 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %913 = tosa.transpose %arg79, %912 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %914 = tosa.reshape %911 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %915 = tosa.reshape %913 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %916 = tosa.matmul %914, %915 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %917 = tosa.reshape %916 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %918 = tosa.reshape %arg80 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %919 = tosa.add %918, %917 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %920 = tosa.reshape %919 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %921 = tosa.reshape %900 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %922 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %923 = tosa.transpose %arg81, %922 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %924 = tosa.reshape %921 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %925 = tosa.reshape %923 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %926 = tosa.matmul %924, %925 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %927 = tosa.reshape %926 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %928 = tosa.reshape %arg82 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %929 = tosa.add %928, %927 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %930 = tosa.reshape %929 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %931 = tosa.reshape %910 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %932 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %933 = tosa.transpose %931, %932 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %934 = tosa.reshape %920 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %935 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %936 = tosa.transpose %934, %935 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %937 = tosa.reshape %930 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %938 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %939 = tosa.transpose %937, %938 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %940 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %941 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %942 = tosa.mul %933, %940 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_201 = tensor.extract_slice %933[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_202 = tensor.extract_slice %933[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %943 = tensor.empty() : tensor<1x12x40x64xf32>
    %944 = linalg.negf ins(%extracted_slice_202 : tensor<1x12x40x64xf32>) outs(%943 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %945 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_203 = tensor.insert_slice %944 into %945[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_204 = tensor.insert_slice %extracted_slice_201 into %inserted_slice_203[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %946 = tosa.mul %inserted_slice_204, %941 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %947 = tosa.add %942, %946 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %948 = tosa.mul %936, %940 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_205 = tensor.extract_slice %936[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_206 = tensor.extract_slice %936[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %949 = tensor.empty() : tensor<1x2x40x64xf32>
    %950 = linalg.negf ins(%extracted_slice_206 : tensor<1x2x40x64xf32>) outs(%949 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %951 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_207 = tensor.insert_slice %950 into %951[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_208 = tensor.insert_slice %extracted_slice_205 into %inserted_slice_207[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %952 = tosa.mul %inserted_slice_208, %941 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %953 = tosa.add %948, %952 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_209 = tensor.extract_slice %953[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_210 = tensor.extract_slice %extracted_slice_209[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %954 = tosa.reshape %extracted_slice_210 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_211 = tensor.extract_slice %954[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_212 = tensor.extract_slice %extracted_slice_211[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %955 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %956 = tosa.add %extracted_slice_212, %955 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %957 = tosa.identity %956 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %958 = tosa.reshape %957 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_213 = tensor.extract_slice %939[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_214 = tensor.extract_slice %extracted_slice_213[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %959 = tosa.reshape %extracted_slice_214 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_215 = tensor.extract_slice %959[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_216 = tensor.extract_slice %extracted_slice_215[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %960 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %961 = tosa.add %extracted_slice_216, %960 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %962 = tosa.identity %961 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %963 = tosa.reshape %962 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_217 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_218 = tensor.extract_slice %extracted_slice_217[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_219 = tensor.extract_slice %extracted_slice_218[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_220 = arith.constant 0.000000e+00 : f32
    %splat_221 = tensor.splat %cst_220 : tensor<40x40xf32>
    %964 = tosa.reshape %extracted_slice_219 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %965 = tosa.add %splat_221, %964 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %966 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %967 = tosa.transpose %958, %966 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %968 = tosa.reshape %947 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %969 = tosa.reshape %967 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %970 = tosa.matmul %968, %969 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_222 = arith.constant 0.0883883461 : f32
    %splat_223 = tensor.splat %cst_222 : tensor<12x40x40xf32>
    %971 = tosa.mul %970, %splat_223 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %972 = tosa.reshape %965 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %973 = tosa.add %971, %972 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %974 = tosa.reduce_max %973 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %975 = tosa.sub %973, %974 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %976 = math.exp %975 : tensor<12x40x40xf32>
    %977 = tosa.reduce_sum %976 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %978 = tosa.log %977 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %979 = tosa.add %974, %978 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %980 = tosa.sub %973, %979 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %981 = math.exp %980 : tensor<12x40x40xf32>
    %982 = tosa.reshape %979 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %983 = tosa.reshape %963 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %984 = tosa.matmul %981, %983 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %985 = tosa.reshape %984 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %986 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %987 = tosa.transpose %985, %986 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %988 = tosa.reshape %987 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %989 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %990 = tosa.transpose %arg83, %989 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %991 = tosa.reshape %988 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_224 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %992 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%991, %990 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_224 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %993 = tosa.reshape %992 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %994 = tosa.add %887, %993 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %995 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_225 = arith.constant 2 : i32
    %996 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%994 : tensor<1x40x1536xf32>) outs(%995 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_225 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %997 = tosa.reduce_sum %996 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %998 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %999 = tosa.reciprocal %998 : (tensor<1xf32>) -> tensor<1xf32>
    %1000 = tosa.reshape %999 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1001 = tosa.mul %1000, %997 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1002 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1003 = tosa.add %1001, %1002 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1004 = tosa.rsqrt %1003 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1005 = tosa.mul %994, %1004 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1006 = tosa.reshape %arg84 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1007 = tosa.mul %1006, %1005 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1008 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1009 = tosa.transpose %arg85, %1008 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1010 = tosa.reshape %1007 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_226 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1011 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1010, %1009 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_226 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1012 = tosa.reshape %1011 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1013 = tosa.sigmoid %1012 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1014 = tosa.mul %1012, %1013 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1015 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1016 = tosa.transpose %arg86, %1015 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1017 = tosa.reshape %1007 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_227 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1018 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1017, %1016 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_227 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1019 = tosa.reshape %1018 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1020 = tosa.mul %1014, %1019 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1021 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1022 = tosa.transpose %arg87, %1021 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %1023 = tosa.reshape %1020 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_228 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1024 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1023, %1022 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_228 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1025 = tosa.reshape %1024 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1026 = tosa.add %994, %1025 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1027 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_229 = arith.constant 2 : i32
    %1028 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1026 : tensor<1x40x1536xf32>) outs(%1027 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_229 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1029 = tosa.reduce_sum %1028 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1030 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1031 = tosa.reciprocal %1030 : (tensor<1xf32>) -> tensor<1xf32>
    %1032 = tosa.reshape %1031 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1033 = tosa.mul %1032, %1029 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1034 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1035 = tosa.add %1033, %1034 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1036 = tosa.rsqrt %1035 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1037 = tosa.mul %1026, %1036 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1038 = tosa.reshape %arg88 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1039 = tosa.mul %1038, %1037 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1040 = tosa.reshape %1039 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1041 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1042 = tosa.transpose %arg89, %1041 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1043 = tosa.reshape %1040 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1044 = tosa.reshape %1042 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %1045 = tosa.matmul %1043, %1044 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %1046 = tosa.reshape %1045 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1047 = tosa.reshape %arg90 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %1048 = tosa.add %1047, %1046 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1049 = tosa.reshape %1048 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1050 = tosa.reshape %1039 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1051 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1052 = tosa.transpose %arg91, %1051 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1053 = tosa.reshape %1050 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1054 = tosa.reshape %1052 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1055 = tosa.matmul %1053, %1054 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1056 = tosa.reshape %1055 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1057 = tosa.reshape %arg92 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1058 = tosa.add %1057, %1056 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1059 = tosa.reshape %1058 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1060 = tosa.reshape %1039 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1061 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1062 = tosa.transpose %arg93, %1061 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1063 = tosa.reshape %1060 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1064 = tosa.reshape %1062 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1065 = tosa.matmul %1063, %1064 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1066 = tosa.reshape %1065 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1067 = tosa.reshape %arg94 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1068 = tosa.add %1067, %1066 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1069 = tosa.reshape %1068 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1070 = tosa.reshape %1049 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %1071 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1072 = tosa.transpose %1070, %1071 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %1073 = tosa.reshape %1059 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1074 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1075 = tosa.transpose %1073, %1074 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1076 = tosa.reshape %1069 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1077 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1078 = tosa.transpose %1076, %1077 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1079 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1080 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1081 = tosa.mul %1072, %1079 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_230 = tensor.extract_slice %1072[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_231 = tensor.extract_slice %1072[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %1082 = tensor.empty() : tensor<1x12x40x64xf32>
    %1083 = linalg.negf ins(%extracted_slice_231 : tensor<1x12x40x64xf32>) outs(%1082 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %1084 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_232 = tensor.insert_slice %1083 into %1084[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_233 = tensor.insert_slice %extracted_slice_230 into %inserted_slice_232[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %1085 = tosa.mul %inserted_slice_233, %1080 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1086 = tosa.add %1081, %1085 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1087 = tosa.mul %1075, %1079 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_234 = tensor.extract_slice %1075[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_235 = tensor.extract_slice %1075[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %1088 = tensor.empty() : tensor<1x2x40x64xf32>
    %1089 = linalg.negf ins(%extracted_slice_235 : tensor<1x2x40x64xf32>) outs(%1088 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %1090 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_236 = tensor.insert_slice %1089 into %1090[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_237 = tensor.insert_slice %extracted_slice_234 into %inserted_slice_236[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %1091 = tosa.mul %inserted_slice_237, %1080 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %1092 = tosa.add %1087, %1091 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_238 = tensor.extract_slice %1092[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_239 = tensor.extract_slice %extracted_slice_238[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1093 = tosa.reshape %extracted_slice_239 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_240 = tensor.extract_slice %1093[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_241 = tensor.extract_slice %extracted_slice_240[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1094 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1095 = tosa.add %extracted_slice_241, %1094 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1096 = tosa.identity %1095 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1097 = tosa.reshape %1096 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_242 = tensor.extract_slice %1078[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_243 = tensor.extract_slice %extracted_slice_242[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1098 = tosa.reshape %extracted_slice_243 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_244 = tensor.extract_slice %1098[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_245 = tensor.extract_slice %extracted_slice_244[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1099 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1100 = tosa.add %extracted_slice_245, %1099 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1101 = tosa.identity %1100 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1102 = tosa.reshape %1101 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_246 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_247 = tensor.extract_slice %extracted_slice_246[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_248 = tensor.extract_slice %extracted_slice_247[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_249 = arith.constant 0.000000e+00 : f32
    %splat_250 = tensor.splat %cst_249 : tensor<40x40xf32>
    %1103 = tosa.reshape %extracted_slice_248 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1104 = tosa.add %splat_250, %1103 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1105 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1106 = tosa.transpose %1097, %1105 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %1107 = tosa.reshape %1086 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1108 = tosa.reshape %1106 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %1109 = tosa.matmul %1107, %1108 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_251 = arith.constant 0.0883883461 : f32
    %splat_252 = tensor.splat %cst_251 : tensor<12x40x40xf32>
    %1110 = tosa.mul %1109, %splat_252 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %1111 = tosa.reshape %1104 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %1112 = tosa.add %1110, %1111 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %1113 = tosa.reduce_max %1112 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1114 = tosa.sub %1112, %1113 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1115 = math.exp %1114 : tensor<12x40x40xf32>
    %1116 = tosa.reduce_sum %1115 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1117 = tosa.log %1116 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1118 = tosa.add %1113, %1117 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1119 = tosa.sub %1112, %1118 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1120 = math.exp %1119 : tensor<12x40x40xf32>
    %1121 = tosa.reshape %1118 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %1122 = tosa.reshape %1102 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1123 = tosa.matmul %1120, %1122 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %1124 = tosa.reshape %1123 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1125 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1126 = tosa.transpose %1124, %1125 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %1127 = tosa.reshape %1126 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %1128 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1129 = tosa.transpose %arg95, %1128 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1130 = tosa.reshape %1127 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_253 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1131 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1130, %1129 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_253 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1132 = tosa.reshape %1131 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1133 = tosa.add %1026, %1132 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1134 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_254 = arith.constant 2 : i32
    %1135 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1133 : tensor<1x40x1536xf32>) outs(%1134 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_254 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1136 = tosa.reduce_sum %1135 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1137 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1138 = tosa.reciprocal %1137 : (tensor<1xf32>) -> tensor<1xf32>
    %1139 = tosa.reshape %1138 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1140 = tosa.mul %1139, %1136 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1141 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1142 = tosa.add %1140, %1141 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1143 = tosa.rsqrt %1142 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1144 = tosa.mul %1133, %1143 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1145 = tosa.reshape %arg96 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1146 = tosa.mul %1145, %1144 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1147 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1148 = tosa.transpose %arg97, %1147 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1149 = tosa.reshape %1146 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_255 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1149, %1148 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_255 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1151 = tosa.reshape %1150 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1152 = tosa.sigmoid %1151 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1153 = tosa.mul %1151, %1152 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1154 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1155 = tosa.transpose %arg98, %1154 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1156 = tosa.reshape %1146 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_256 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1157 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1156, %1155 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_256 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1158 = tosa.reshape %1157 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1159 = tosa.mul %1153, %1158 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1160 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1161 = tosa.transpose %arg99, %1160 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %1162 = tosa.reshape %1159 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_257 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1163 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1162, %1161 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_257 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1164 = tosa.reshape %1163 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1165 = tosa.add %1133, %1164 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1166 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_258 = arith.constant 2 : i32
    %1167 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1165 : tensor<1x40x1536xf32>) outs(%1166 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_258 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1168 = tosa.reduce_sum %1167 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1169 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1170 = tosa.reciprocal %1169 : (tensor<1xf32>) -> tensor<1xf32>
    %1171 = tosa.reshape %1170 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1172 = tosa.mul %1171, %1168 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1173 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1174 = tosa.add %1172, %1173 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1175 = tosa.rsqrt %1174 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1176 = tosa.mul %1165, %1175 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1177 = tosa.reshape %arg100 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1178 = tosa.mul %1177, %1176 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1179 = tosa.reshape %1178 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1180 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1181 = tosa.transpose %arg101, %1180 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1182 = tosa.reshape %1179 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1183 = tosa.reshape %1181 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %1184 = tosa.matmul %1182, %1183 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %1185 = tosa.reshape %1184 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1186 = tosa.reshape %arg102 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %1187 = tosa.add %1186, %1185 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1188 = tosa.reshape %1187 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1189 = tosa.reshape %1178 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1190 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1191 = tosa.transpose %arg103, %1190 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1192 = tosa.reshape %1189 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1193 = tosa.reshape %1191 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1194 = tosa.matmul %1192, %1193 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1195 = tosa.reshape %1194 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1196 = tosa.reshape %arg104 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1197 = tosa.add %1196, %1195 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1198 = tosa.reshape %1197 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1199 = tosa.reshape %1178 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1200 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1201 = tosa.transpose %arg105, %1200 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1202 = tosa.reshape %1199 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1203 = tosa.reshape %1201 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1204 = tosa.matmul %1202, %1203 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1205 = tosa.reshape %1204 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1206 = tosa.reshape %arg106 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1207 = tosa.add %1206, %1205 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1208 = tosa.reshape %1207 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1209 = tosa.reshape %1188 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %1210 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1211 = tosa.transpose %1209, %1210 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %1212 = tosa.reshape %1198 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1213 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1214 = tosa.transpose %1212, %1213 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1215 = tosa.reshape %1208 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1216 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1217 = tosa.transpose %1215, %1216 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1218 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1219 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1220 = tosa.mul %1211, %1218 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_259 = tensor.extract_slice %1211[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_260 = tensor.extract_slice %1211[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %1221 = tensor.empty() : tensor<1x12x40x64xf32>
    %1222 = linalg.negf ins(%extracted_slice_260 : tensor<1x12x40x64xf32>) outs(%1221 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %1223 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_261 = tensor.insert_slice %1222 into %1223[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_262 = tensor.insert_slice %extracted_slice_259 into %inserted_slice_261[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %1224 = tosa.mul %inserted_slice_262, %1219 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1225 = tosa.add %1220, %1224 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1226 = tosa.mul %1214, %1218 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_263 = tensor.extract_slice %1214[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_264 = tensor.extract_slice %1214[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %1227 = tensor.empty() : tensor<1x2x40x64xf32>
    %1228 = linalg.negf ins(%extracted_slice_264 : tensor<1x2x40x64xf32>) outs(%1227 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %1229 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_265 = tensor.insert_slice %1228 into %1229[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_266 = tensor.insert_slice %extracted_slice_263 into %inserted_slice_265[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %1230 = tosa.mul %inserted_slice_266, %1219 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %1231 = tosa.add %1226, %1230 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_267 = tensor.extract_slice %1231[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_268 = tensor.extract_slice %extracted_slice_267[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1232 = tosa.reshape %extracted_slice_268 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_269 = tensor.extract_slice %1232[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_270 = tensor.extract_slice %extracted_slice_269[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1233 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1234 = tosa.add %extracted_slice_270, %1233 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1235 = tosa.identity %1234 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1236 = tosa.reshape %1235 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_271 = tensor.extract_slice %1217[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_272 = tensor.extract_slice %extracted_slice_271[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1237 = tosa.reshape %extracted_slice_272 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_273 = tensor.extract_slice %1237[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_274 = tensor.extract_slice %extracted_slice_273[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1238 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1239 = tosa.add %extracted_slice_274, %1238 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1240 = tosa.identity %1239 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1241 = tosa.reshape %1240 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_275 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_276 = tensor.extract_slice %extracted_slice_275[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_277 = tensor.extract_slice %extracted_slice_276[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_278 = arith.constant 0.000000e+00 : f32
    %splat_279 = tensor.splat %cst_278 : tensor<40x40xf32>
    %1242 = tosa.reshape %extracted_slice_277 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1243 = tosa.add %splat_279, %1242 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1244 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1245 = tosa.transpose %1236, %1244 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %1246 = tosa.reshape %1225 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1247 = tosa.reshape %1245 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %1248 = tosa.matmul %1246, %1247 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_280 = arith.constant 0.0883883461 : f32
    %splat_281 = tensor.splat %cst_280 : tensor<12x40x40xf32>
    %1249 = tosa.mul %1248, %splat_281 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %1250 = tosa.reshape %1243 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %1251 = tosa.add %1249, %1250 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %1252 = tosa.reduce_max %1251 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1253 = tosa.sub %1251, %1252 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1254 = math.exp %1253 : tensor<12x40x40xf32>
    %1255 = tosa.reduce_sum %1254 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1256 = tosa.log %1255 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1257 = tosa.add %1252, %1256 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1258 = tosa.sub %1251, %1257 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1259 = math.exp %1258 : tensor<12x40x40xf32>
    %1260 = tosa.reshape %1257 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %1261 = tosa.reshape %1241 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1262 = tosa.matmul %1259, %1261 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %1263 = tosa.reshape %1262 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1264 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1265 = tosa.transpose %1263, %1264 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %1266 = tosa.reshape %1265 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %1267 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1268 = tosa.transpose %arg107, %1267 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1269 = tosa.reshape %1266 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_282 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1270 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1269, %1268 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_282 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1271 = tosa.reshape %1270 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1272 = tosa.add %1165, %1271 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1273 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_283 = arith.constant 2 : i32
    %1274 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1272 : tensor<1x40x1536xf32>) outs(%1273 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_283 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1275 = tosa.reduce_sum %1274 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1276 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1277 = tosa.reciprocal %1276 : (tensor<1xf32>) -> tensor<1xf32>
    %1278 = tosa.reshape %1277 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1279 = tosa.mul %1278, %1275 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1280 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1281 = tosa.add %1279, %1280 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1282 = tosa.rsqrt %1281 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1283 = tosa.mul %1272, %1282 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1284 = tosa.reshape %arg108 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1285 = tosa.mul %1284, %1283 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1286 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1287 = tosa.transpose %arg109, %1286 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1288 = tosa.reshape %1285 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_284 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1289 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1288, %1287 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_284 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1290 = tosa.reshape %1289 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1291 = tosa.sigmoid %1290 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1292 = tosa.mul %1290, %1291 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1293 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1294 = tosa.transpose %arg110, %1293 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1295 = tosa.reshape %1285 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_285 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1296 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1295, %1294 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_285 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1297 = tosa.reshape %1296 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1298 = tosa.mul %1292, %1297 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1299 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1300 = tosa.transpose %arg111, %1299 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %1301 = tosa.reshape %1298 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_286 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1302 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1301, %1300 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_286 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1303 = tosa.reshape %1302 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1304 = tosa.add %1272, %1303 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1305 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_287 = arith.constant 2 : i32
    %1306 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1304 : tensor<1x40x1536xf32>) outs(%1305 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_287 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1307 = tosa.reduce_sum %1306 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1308 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1309 = tosa.reciprocal %1308 : (tensor<1xf32>) -> tensor<1xf32>
    %1310 = tosa.reshape %1309 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1311 = tosa.mul %1310, %1307 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1312 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1313 = tosa.add %1311, %1312 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1314 = tosa.rsqrt %1313 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1315 = tosa.mul %1304, %1314 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1316 = tosa.reshape %arg112 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1317 = tosa.mul %1316, %1315 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1318 = tosa.reshape %1317 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1319 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1320 = tosa.transpose %arg113, %1319 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1321 = tosa.reshape %1318 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1322 = tosa.reshape %1320 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %1323 = tosa.matmul %1321, %1322 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %1324 = tosa.reshape %1323 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1325 = tosa.reshape %arg114 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %1326 = tosa.add %1325, %1324 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1327 = tosa.reshape %1326 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1328 = tosa.reshape %1317 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1329 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1330 = tosa.transpose %arg115, %1329 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1331 = tosa.reshape %1328 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1332 = tosa.reshape %1330 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1333 = tosa.matmul %1331, %1332 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1334 = tosa.reshape %1333 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1335 = tosa.reshape %arg116 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1336 = tosa.add %1335, %1334 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1337 = tosa.reshape %1336 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1338 = tosa.reshape %1317 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1339 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1340 = tosa.transpose %arg117, %1339 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1341 = tosa.reshape %1338 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1342 = tosa.reshape %1340 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1343 = tosa.matmul %1341, %1342 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1344 = tosa.reshape %1343 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1345 = tosa.reshape %arg118 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1346 = tosa.add %1345, %1344 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1347 = tosa.reshape %1346 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1348 = tosa.reshape %1327 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %1349 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1350 = tosa.transpose %1348, %1349 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %1351 = tosa.reshape %1337 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1352 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1353 = tosa.transpose %1351, %1352 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1354 = tosa.reshape %1347 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1355 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1356 = tosa.transpose %1354, %1355 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1357 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1358 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1359 = tosa.mul %1350, %1357 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_288 = tensor.extract_slice %1350[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_289 = tensor.extract_slice %1350[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %1360 = tensor.empty() : tensor<1x12x40x64xf32>
    %1361 = linalg.negf ins(%extracted_slice_289 : tensor<1x12x40x64xf32>) outs(%1360 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %1362 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_290 = tensor.insert_slice %1361 into %1362[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_291 = tensor.insert_slice %extracted_slice_288 into %inserted_slice_290[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %1363 = tosa.mul %inserted_slice_291, %1358 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1364 = tosa.add %1359, %1363 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1365 = tosa.mul %1353, %1357 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_292 = tensor.extract_slice %1353[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_293 = tensor.extract_slice %1353[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %1366 = tensor.empty() : tensor<1x2x40x64xf32>
    %1367 = linalg.negf ins(%extracted_slice_293 : tensor<1x2x40x64xf32>) outs(%1366 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %1368 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_294 = tensor.insert_slice %1367 into %1368[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_295 = tensor.insert_slice %extracted_slice_292 into %inserted_slice_294[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %1369 = tosa.mul %inserted_slice_295, %1358 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %1370 = tosa.add %1365, %1369 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_296 = tensor.extract_slice %1370[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_297 = tensor.extract_slice %extracted_slice_296[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1371 = tosa.reshape %extracted_slice_297 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_298 = tensor.extract_slice %1371[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_299 = tensor.extract_slice %extracted_slice_298[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1372 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1373 = tosa.add %extracted_slice_299, %1372 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1374 = tosa.identity %1373 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1375 = tosa.reshape %1374 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_300 = tensor.extract_slice %1356[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_301 = tensor.extract_slice %extracted_slice_300[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1376 = tosa.reshape %extracted_slice_301 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_302 = tensor.extract_slice %1376[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_303 = tensor.extract_slice %extracted_slice_302[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1377 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1378 = tosa.add %extracted_slice_303, %1377 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1379 = tosa.identity %1378 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1380 = tosa.reshape %1379 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_304 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_305 = tensor.extract_slice %extracted_slice_304[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_306 = tensor.extract_slice %extracted_slice_305[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_307 = arith.constant 0.000000e+00 : f32
    %splat_308 = tensor.splat %cst_307 : tensor<40x40xf32>
    %1381 = tosa.reshape %extracted_slice_306 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1382 = tosa.add %splat_308, %1381 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1383 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1384 = tosa.transpose %1375, %1383 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %1385 = tosa.reshape %1364 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1386 = tosa.reshape %1384 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %1387 = tosa.matmul %1385, %1386 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_309 = arith.constant 0.0883883461 : f32
    %splat_310 = tensor.splat %cst_309 : tensor<12x40x40xf32>
    %1388 = tosa.mul %1387, %splat_310 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %1389 = tosa.reshape %1382 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %1390 = tosa.add %1388, %1389 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %1391 = tosa.reduce_max %1390 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1392 = tosa.sub %1390, %1391 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1393 = math.exp %1392 : tensor<12x40x40xf32>
    %1394 = tosa.reduce_sum %1393 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1395 = tosa.log %1394 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1396 = tosa.add %1391, %1395 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1397 = tosa.sub %1390, %1396 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1398 = math.exp %1397 : tensor<12x40x40xf32>
    %1399 = tosa.reshape %1396 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %1400 = tosa.reshape %1380 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1401 = tosa.matmul %1398, %1400 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %1402 = tosa.reshape %1401 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1403 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1404 = tosa.transpose %1402, %1403 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %1405 = tosa.reshape %1404 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %1406 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1407 = tosa.transpose %arg119, %1406 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1408 = tosa.reshape %1405 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_311 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1409 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1408, %1407 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_311 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1410 = tosa.reshape %1409 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1411 = tosa.add %1304, %1410 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1412 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_312 = arith.constant 2 : i32
    %1413 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1411 : tensor<1x40x1536xf32>) outs(%1412 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_312 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1414 = tosa.reduce_sum %1413 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1415 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1416 = tosa.reciprocal %1415 : (tensor<1xf32>) -> tensor<1xf32>
    %1417 = tosa.reshape %1416 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1418 = tosa.mul %1417, %1414 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1419 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1420 = tosa.add %1418, %1419 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1421 = tosa.rsqrt %1420 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1422 = tosa.mul %1411, %1421 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1423 = tosa.reshape %arg120 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1424 = tosa.mul %1423, %1422 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1425 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1426 = tosa.transpose %arg121, %1425 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1427 = tosa.reshape %1424 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_313 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1428 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1427, %1426 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_313 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1429 = tosa.reshape %1428 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1430 = tosa.sigmoid %1429 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1431 = tosa.mul %1429, %1430 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1432 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1433 = tosa.transpose %arg122, %1432 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1434 = tosa.reshape %1424 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_314 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1435 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1434, %1433 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_314 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1436 = tosa.reshape %1435 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1437 = tosa.mul %1431, %1436 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1438 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1439 = tosa.transpose %arg123, %1438 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %1440 = tosa.reshape %1437 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_315 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1441 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1440, %1439 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_315 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1442 = tosa.reshape %1441 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1443 = tosa.add %1411, %1442 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1444 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_316 = arith.constant 2 : i32
    %1445 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1443 : tensor<1x40x1536xf32>) outs(%1444 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_316 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1446 = tosa.reduce_sum %1445 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1447 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1448 = tosa.reciprocal %1447 : (tensor<1xf32>) -> tensor<1xf32>
    %1449 = tosa.reshape %1448 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1450 = tosa.mul %1449, %1446 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1451 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1452 = tosa.add %1450, %1451 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1453 = tosa.rsqrt %1452 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1454 = tosa.mul %1443, %1453 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1455 = tosa.reshape %arg124 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1456 = tosa.mul %1455, %1454 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1457 = tosa.reshape %1456 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1458 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1459 = tosa.transpose %arg125, %1458 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1460 = tosa.reshape %1457 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1461 = tosa.reshape %1459 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %1462 = tosa.matmul %1460, %1461 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %1463 = tosa.reshape %1462 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1464 = tosa.reshape %arg126 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %1465 = tosa.add %1464, %1463 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1466 = tosa.reshape %1465 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1467 = tosa.reshape %1456 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1468 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1469 = tosa.transpose %arg127, %1468 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1470 = tosa.reshape %1467 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1471 = tosa.reshape %1469 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1472 = tosa.matmul %1470, %1471 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1473 = tosa.reshape %1472 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1474 = tosa.reshape %arg128 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1475 = tosa.add %1474, %1473 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1476 = tosa.reshape %1475 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1477 = tosa.reshape %1456 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1478 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1479 = tosa.transpose %arg129, %1478 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1480 = tosa.reshape %1477 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1481 = tosa.reshape %1479 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1482 = tosa.matmul %1480, %1481 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1483 = tosa.reshape %1482 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1484 = tosa.reshape %arg130 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1485 = tosa.add %1484, %1483 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1486 = tosa.reshape %1485 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1487 = tosa.reshape %1466 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %1488 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1489 = tosa.transpose %1487, %1488 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %1490 = tosa.reshape %1476 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1491 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1492 = tosa.transpose %1490, %1491 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1493 = tosa.reshape %1486 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1494 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1495 = tosa.transpose %1493, %1494 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1496 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1497 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1498 = tosa.mul %1489, %1496 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_317 = tensor.extract_slice %1489[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_318 = tensor.extract_slice %1489[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %1499 = tensor.empty() : tensor<1x12x40x64xf32>
    %1500 = linalg.negf ins(%extracted_slice_318 : tensor<1x12x40x64xf32>) outs(%1499 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %1501 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_319 = tensor.insert_slice %1500 into %1501[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_320 = tensor.insert_slice %extracted_slice_317 into %inserted_slice_319[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %1502 = tosa.mul %inserted_slice_320, %1497 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1503 = tosa.add %1498, %1502 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1504 = tosa.mul %1492, %1496 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_321 = tensor.extract_slice %1492[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_322 = tensor.extract_slice %1492[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %1505 = tensor.empty() : tensor<1x2x40x64xf32>
    %1506 = linalg.negf ins(%extracted_slice_322 : tensor<1x2x40x64xf32>) outs(%1505 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %1507 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_323 = tensor.insert_slice %1506 into %1507[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_324 = tensor.insert_slice %extracted_slice_321 into %inserted_slice_323[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %1508 = tosa.mul %inserted_slice_324, %1497 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %1509 = tosa.add %1504, %1508 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_325 = tensor.extract_slice %1509[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_326 = tensor.extract_slice %extracted_slice_325[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1510 = tosa.reshape %extracted_slice_326 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_327 = tensor.extract_slice %1510[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_328 = tensor.extract_slice %extracted_slice_327[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1511 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1512 = tosa.add %extracted_slice_328, %1511 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1513 = tosa.identity %1512 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1514 = tosa.reshape %1513 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_329 = tensor.extract_slice %1495[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_330 = tensor.extract_slice %extracted_slice_329[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1515 = tosa.reshape %extracted_slice_330 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_331 = tensor.extract_slice %1515[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_332 = tensor.extract_slice %extracted_slice_331[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1516 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1517 = tosa.add %extracted_slice_332, %1516 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1518 = tosa.identity %1517 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1519 = tosa.reshape %1518 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_333 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_334 = tensor.extract_slice %extracted_slice_333[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_335 = tensor.extract_slice %extracted_slice_334[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_336 = arith.constant 0.000000e+00 : f32
    %splat_337 = tensor.splat %cst_336 : tensor<40x40xf32>
    %1520 = tosa.reshape %extracted_slice_335 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1521 = tosa.add %splat_337, %1520 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1522 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1523 = tosa.transpose %1514, %1522 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %1524 = tosa.reshape %1503 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1525 = tosa.reshape %1523 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %1526 = tosa.matmul %1524, %1525 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_338 = arith.constant 0.0883883461 : f32
    %splat_339 = tensor.splat %cst_338 : tensor<12x40x40xf32>
    %1527 = tosa.mul %1526, %splat_339 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %1528 = tosa.reshape %1521 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %1529 = tosa.add %1527, %1528 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %1530 = tosa.reduce_max %1529 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1531 = tosa.sub %1529, %1530 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1532 = math.exp %1531 : tensor<12x40x40xf32>
    %1533 = tosa.reduce_sum %1532 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1534 = tosa.log %1533 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1535 = tosa.add %1530, %1534 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1536 = tosa.sub %1529, %1535 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1537 = math.exp %1536 : tensor<12x40x40xf32>
    %1538 = tosa.reshape %1535 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %1539 = tosa.reshape %1519 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1540 = tosa.matmul %1537, %1539 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %1541 = tosa.reshape %1540 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1542 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1543 = tosa.transpose %1541, %1542 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %1544 = tosa.reshape %1543 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %1545 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1546 = tosa.transpose %arg131, %1545 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1547 = tosa.reshape %1544 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_340 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1548 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1547, %1546 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_340 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1549 = tosa.reshape %1548 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1550 = tosa.add %1443, %1549 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1551 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_341 = arith.constant 2 : i32
    %1552 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1550 : tensor<1x40x1536xf32>) outs(%1551 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_341 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1553 = tosa.reduce_sum %1552 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1554 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1555 = tosa.reciprocal %1554 : (tensor<1xf32>) -> tensor<1xf32>
    %1556 = tosa.reshape %1555 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1557 = tosa.mul %1556, %1553 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1558 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1559 = tosa.add %1557, %1558 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1560 = tosa.rsqrt %1559 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1561 = tosa.mul %1550, %1560 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1562 = tosa.reshape %arg132 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1563 = tosa.mul %1562, %1561 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1564 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1565 = tosa.transpose %arg133, %1564 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1566 = tosa.reshape %1563 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_342 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1567 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1566, %1565 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_342 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1568 = tosa.reshape %1567 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1569 = tosa.sigmoid %1568 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1570 = tosa.mul %1568, %1569 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1571 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1572 = tosa.transpose %arg134, %1571 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1573 = tosa.reshape %1563 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_343 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1574 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1573, %1572 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_343 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1575 = tosa.reshape %1574 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1576 = tosa.mul %1570, %1575 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1577 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1578 = tosa.transpose %arg135, %1577 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %1579 = tosa.reshape %1576 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_344 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1580 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1579, %1578 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_344 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1581 = tosa.reshape %1580 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1582 = tosa.add %1550, %1581 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1583 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_345 = arith.constant 2 : i32
    %1584 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1582 : tensor<1x40x1536xf32>) outs(%1583 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_345 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1585 = tosa.reduce_sum %1584 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1586 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1587 = tosa.reciprocal %1586 : (tensor<1xf32>) -> tensor<1xf32>
    %1588 = tosa.reshape %1587 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1589 = tosa.mul %1588, %1585 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1590 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1591 = tosa.add %1589, %1590 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1592 = tosa.rsqrt %1591 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1593 = tosa.mul %1582, %1592 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1594 = tosa.reshape %arg136 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1595 = tosa.mul %1594, %1593 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1596 = tosa.reshape %1595 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1597 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1598 = tosa.transpose %arg137, %1597 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1599 = tosa.reshape %1596 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1600 = tosa.reshape %1598 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %1601 = tosa.matmul %1599, %1600 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %1602 = tosa.reshape %1601 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1603 = tosa.reshape %arg138 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %1604 = tosa.add %1603, %1602 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1605 = tosa.reshape %1604 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1606 = tosa.reshape %1595 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1607 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1608 = tosa.transpose %arg139, %1607 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1609 = tosa.reshape %1606 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1610 = tosa.reshape %1608 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1611 = tosa.matmul %1609, %1610 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1612 = tosa.reshape %1611 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1613 = tosa.reshape %arg140 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1614 = tosa.add %1613, %1612 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1615 = tosa.reshape %1614 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1616 = tosa.reshape %1595 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1617 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1618 = tosa.transpose %arg141, %1617 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1619 = tosa.reshape %1616 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1620 = tosa.reshape %1618 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1621 = tosa.matmul %1619, %1620 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1622 = tosa.reshape %1621 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1623 = tosa.reshape %arg142 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1624 = tosa.add %1623, %1622 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1625 = tosa.reshape %1624 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1626 = tosa.reshape %1605 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %1627 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1628 = tosa.transpose %1626, %1627 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %1629 = tosa.reshape %1615 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1630 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1631 = tosa.transpose %1629, %1630 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1632 = tosa.reshape %1625 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1633 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1634 = tosa.transpose %1632, %1633 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1635 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1636 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1637 = tosa.mul %1628, %1635 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_346 = tensor.extract_slice %1628[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_347 = tensor.extract_slice %1628[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %1638 = tensor.empty() : tensor<1x12x40x64xf32>
    %1639 = linalg.negf ins(%extracted_slice_347 : tensor<1x12x40x64xf32>) outs(%1638 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %1640 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_348 = tensor.insert_slice %1639 into %1640[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_349 = tensor.insert_slice %extracted_slice_346 into %inserted_slice_348[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %1641 = tosa.mul %inserted_slice_349, %1636 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1642 = tosa.add %1637, %1641 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1643 = tosa.mul %1631, %1635 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_350 = tensor.extract_slice %1631[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_351 = tensor.extract_slice %1631[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %1644 = tensor.empty() : tensor<1x2x40x64xf32>
    %1645 = linalg.negf ins(%extracted_slice_351 : tensor<1x2x40x64xf32>) outs(%1644 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %1646 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_352 = tensor.insert_slice %1645 into %1646[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_353 = tensor.insert_slice %extracted_slice_350 into %inserted_slice_352[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %1647 = tosa.mul %inserted_slice_353, %1636 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %1648 = tosa.add %1643, %1647 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_354 = tensor.extract_slice %1648[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_355 = tensor.extract_slice %extracted_slice_354[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1649 = tosa.reshape %extracted_slice_355 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_356 = tensor.extract_slice %1649[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_357 = tensor.extract_slice %extracted_slice_356[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1650 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1651 = tosa.add %extracted_slice_357, %1650 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1652 = tosa.identity %1651 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1653 = tosa.reshape %1652 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_358 = tensor.extract_slice %1634[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_359 = tensor.extract_slice %extracted_slice_358[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1654 = tosa.reshape %extracted_slice_359 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_360 = tensor.extract_slice %1654[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_361 = tensor.extract_slice %extracted_slice_360[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1655 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1656 = tosa.add %extracted_slice_361, %1655 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1657 = tosa.identity %1656 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1658 = tosa.reshape %1657 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_362 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_363 = tensor.extract_slice %extracted_slice_362[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_364 = tensor.extract_slice %extracted_slice_363[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_365 = arith.constant 0.000000e+00 : f32
    %splat_366 = tensor.splat %cst_365 : tensor<40x40xf32>
    %1659 = tosa.reshape %extracted_slice_364 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1660 = tosa.add %splat_366, %1659 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1661 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1662 = tosa.transpose %1653, %1661 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %1663 = tosa.reshape %1642 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1664 = tosa.reshape %1662 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %1665 = tosa.matmul %1663, %1664 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_367 = arith.constant 0.0883883461 : f32
    %splat_368 = tensor.splat %cst_367 : tensor<12x40x40xf32>
    %1666 = tosa.mul %1665, %splat_368 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %1667 = tosa.reshape %1660 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %1668 = tosa.add %1666, %1667 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %1669 = tosa.reduce_max %1668 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1670 = tosa.sub %1668, %1669 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1671 = math.exp %1670 : tensor<12x40x40xf32>
    %1672 = tosa.reduce_sum %1671 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1673 = tosa.log %1672 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1674 = tosa.add %1669, %1673 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1675 = tosa.sub %1668, %1674 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1676 = math.exp %1675 : tensor<12x40x40xf32>
    %1677 = tosa.reshape %1674 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %1678 = tosa.reshape %1658 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1679 = tosa.matmul %1676, %1678 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %1680 = tosa.reshape %1679 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1681 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1682 = tosa.transpose %1680, %1681 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %1683 = tosa.reshape %1682 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %1684 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1685 = tosa.transpose %arg143, %1684 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1686 = tosa.reshape %1683 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_369 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1687 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1686, %1685 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_369 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1688 = tosa.reshape %1687 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1689 = tosa.add %1582, %1688 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1690 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_370 = arith.constant 2 : i32
    %1691 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1689 : tensor<1x40x1536xf32>) outs(%1690 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_370 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1692 = tosa.reduce_sum %1691 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1693 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1694 = tosa.reciprocal %1693 : (tensor<1xf32>) -> tensor<1xf32>
    %1695 = tosa.reshape %1694 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1696 = tosa.mul %1695, %1692 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1697 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1698 = tosa.add %1696, %1697 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1699 = tosa.rsqrt %1698 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1700 = tosa.mul %1689, %1699 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1701 = tosa.reshape %arg144 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1702 = tosa.mul %1701, %1700 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1703 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1704 = tosa.transpose %arg145, %1703 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1705 = tosa.reshape %1702 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_371 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1706 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1705, %1704 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_371 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1707 = tosa.reshape %1706 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1708 = tosa.sigmoid %1707 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1709 = tosa.mul %1707, %1708 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1710 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1711 = tosa.transpose %arg146, %1710 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1712 = tosa.reshape %1702 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_372 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1713 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1712, %1711 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_372 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1714 = tosa.reshape %1713 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1715 = tosa.mul %1709, %1714 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1716 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1717 = tosa.transpose %arg147, %1716 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %1718 = tosa.reshape %1715 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_373 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1719 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1718, %1717 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_373 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1720 = tosa.reshape %1719 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1721 = tosa.add %1689, %1720 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1722 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_374 = arith.constant 2 : i32
    %1723 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1721 : tensor<1x40x1536xf32>) outs(%1722 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_374 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1724 = tosa.reduce_sum %1723 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1725 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1726 = tosa.reciprocal %1725 : (tensor<1xf32>) -> tensor<1xf32>
    %1727 = tosa.reshape %1726 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1728 = tosa.mul %1727, %1724 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1729 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1730 = tosa.add %1728, %1729 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1731 = tosa.rsqrt %1730 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1732 = tosa.mul %1721, %1731 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1733 = tosa.reshape %arg148 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1734 = tosa.mul %1733, %1732 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1735 = tosa.reshape %1734 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1736 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1737 = tosa.transpose %arg149, %1736 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1738 = tosa.reshape %1735 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1739 = tosa.reshape %1737 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %1740 = tosa.matmul %1738, %1739 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %1741 = tosa.reshape %1740 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1742 = tosa.reshape %arg150 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %1743 = tosa.add %1742, %1741 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1744 = tosa.reshape %1743 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1745 = tosa.reshape %1734 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1746 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1747 = tosa.transpose %arg151, %1746 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1748 = tosa.reshape %1745 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1749 = tosa.reshape %1747 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1750 = tosa.matmul %1748, %1749 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1751 = tosa.reshape %1750 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1752 = tosa.reshape %arg152 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1753 = tosa.add %1752, %1751 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1754 = tosa.reshape %1753 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1755 = tosa.reshape %1734 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1756 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1757 = tosa.transpose %arg153, %1756 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1758 = tosa.reshape %1755 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1759 = tosa.reshape %1757 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1760 = tosa.matmul %1758, %1759 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1761 = tosa.reshape %1760 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1762 = tosa.reshape %arg154 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1763 = tosa.add %1762, %1761 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1764 = tosa.reshape %1763 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1765 = tosa.reshape %1744 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %1766 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1767 = tosa.transpose %1765, %1766 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %1768 = tosa.reshape %1754 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1769 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1770 = tosa.transpose %1768, %1769 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1771 = tosa.reshape %1764 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1772 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1773 = tosa.transpose %1771, %1772 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1774 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1775 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1776 = tosa.mul %1767, %1774 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_375 = tensor.extract_slice %1767[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_376 = tensor.extract_slice %1767[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %1777 = tensor.empty() : tensor<1x12x40x64xf32>
    %1778 = linalg.negf ins(%extracted_slice_376 : tensor<1x12x40x64xf32>) outs(%1777 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %1779 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_377 = tensor.insert_slice %1778 into %1779[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_378 = tensor.insert_slice %extracted_slice_375 into %inserted_slice_377[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %1780 = tosa.mul %inserted_slice_378, %1775 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1781 = tosa.add %1776, %1780 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1782 = tosa.mul %1770, %1774 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_379 = tensor.extract_slice %1770[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_380 = tensor.extract_slice %1770[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %1783 = tensor.empty() : tensor<1x2x40x64xf32>
    %1784 = linalg.negf ins(%extracted_slice_380 : tensor<1x2x40x64xf32>) outs(%1783 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %1785 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_381 = tensor.insert_slice %1784 into %1785[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_382 = tensor.insert_slice %extracted_slice_379 into %inserted_slice_381[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %1786 = tosa.mul %inserted_slice_382, %1775 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %1787 = tosa.add %1782, %1786 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_383 = tensor.extract_slice %1787[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_384 = tensor.extract_slice %extracted_slice_383[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1788 = tosa.reshape %extracted_slice_384 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_385 = tensor.extract_slice %1788[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_386 = tensor.extract_slice %extracted_slice_385[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1789 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1790 = tosa.add %extracted_slice_386, %1789 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1791 = tosa.identity %1790 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1792 = tosa.reshape %1791 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_387 = tensor.extract_slice %1773[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_388 = tensor.extract_slice %extracted_slice_387[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1793 = tosa.reshape %extracted_slice_388 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_389 = tensor.extract_slice %1793[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_390 = tensor.extract_slice %extracted_slice_389[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1794 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1795 = tosa.add %extracted_slice_390, %1794 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1796 = tosa.identity %1795 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1797 = tosa.reshape %1796 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_391 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_392 = tensor.extract_slice %extracted_slice_391[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_393 = tensor.extract_slice %extracted_slice_392[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_394 = arith.constant 0.000000e+00 : f32
    %splat_395 = tensor.splat %cst_394 : tensor<40x40xf32>
    %1798 = tosa.reshape %extracted_slice_393 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1799 = tosa.add %splat_395, %1798 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1800 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1801 = tosa.transpose %1792, %1800 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %1802 = tosa.reshape %1781 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1803 = tosa.reshape %1801 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %1804 = tosa.matmul %1802, %1803 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_396 = arith.constant 0.0883883461 : f32
    %splat_397 = tensor.splat %cst_396 : tensor<12x40x40xf32>
    %1805 = tosa.mul %1804, %splat_397 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %1806 = tosa.reshape %1799 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %1807 = tosa.add %1805, %1806 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %1808 = tosa.reduce_max %1807 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1809 = tosa.sub %1807, %1808 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1810 = math.exp %1809 : tensor<12x40x40xf32>
    %1811 = tosa.reduce_sum %1810 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1812 = tosa.log %1811 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1813 = tosa.add %1808, %1812 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1814 = tosa.sub %1807, %1813 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1815 = math.exp %1814 : tensor<12x40x40xf32>
    %1816 = tosa.reshape %1813 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %1817 = tosa.reshape %1797 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1818 = tosa.matmul %1815, %1817 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %1819 = tosa.reshape %1818 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1820 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1821 = tosa.transpose %1819, %1820 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %1822 = tosa.reshape %1821 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %1823 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1824 = tosa.transpose %arg155, %1823 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1825 = tosa.reshape %1822 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_398 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1826 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1825, %1824 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_398 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1827 = tosa.reshape %1826 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1828 = tosa.add %1721, %1827 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1829 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_399 = arith.constant 2 : i32
    %1830 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1828 : tensor<1x40x1536xf32>) outs(%1829 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_399 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1831 = tosa.reduce_sum %1830 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1832 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1833 = tosa.reciprocal %1832 : (tensor<1xf32>) -> tensor<1xf32>
    %1834 = tosa.reshape %1833 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1835 = tosa.mul %1834, %1831 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1836 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1837 = tosa.add %1835, %1836 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1838 = tosa.rsqrt %1837 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1839 = tosa.mul %1828, %1838 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1840 = tosa.reshape %arg156 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1841 = tosa.mul %1840, %1839 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1842 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1843 = tosa.transpose %arg157, %1842 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1844 = tosa.reshape %1841 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_400 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1845 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1844, %1843 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_400 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1846 = tosa.reshape %1845 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1847 = tosa.sigmoid %1846 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1848 = tosa.mul %1846, %1847 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1849 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1850 = tosa.transpose %arg158, %1849 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1851 = tosa.reshape %1841 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_401 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1852 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1851, %1850 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_401 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1853 = tosa.reshape %1852 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1854 = tosa.mul %1848, %1853 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1855 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1856 = tosa.transpose %arg159, %1855 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %1857 = tosa.reshape %1854 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_402 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1858 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1857, %1856 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_402 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1859 = tosa.reshape %1858 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1860 = tosa.add %1828, %1859 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1861 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_403 = arith.constant 2 : i32
    %1862 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1860 : tensor<1x40x1536xf32>) outs(%1861 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_403 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1863 = tosa.reduce_sum %1862 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1864 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1865 = tosa.reciprocal %1864 : (tensor<1xf32>) -> tensor<1xf32>
    %1866 = tosa.reshape %1865 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1867 = tosa.mul %1866, %1863 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1868 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1869 = tosa.add %1867, %1868 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1870 = tosa.rsqrt %1869 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1871 = tosa.mul %1860, %1870 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1872 = tosa.reshape %arg160 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1873 = tosa.mul %1872, %1871 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1874 = tosa.reshape %1873 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1875 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1876 = tosa.transpose %arg161, %1875 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1877 = tosa.reshape %1874 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1878 = tosa.reshape %1876 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %1879 = tosa.matmul %1877, %1878 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %1880 = tosa.reshape %1879 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1881 = tosa.reshape %arg162 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %1882 = tosa.add %1881, %1880 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1883 = tosa.reshape %1882 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1884 = tosa.reshape %1873 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1885 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1886 = tosa.transpose %arg163, %1885 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1887 = tosa.reshape %1884 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1888 = tosa.reshape %1886 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1889 = tosa.matmul %1887, %1888 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1890 = tosa.reshape %1889 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1891 = tosa.reshape %arg164 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1892 = tosa.add %1891, %1890 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1893 = tosa.reshape %1892 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1894 = tosa.reshape %1873 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %1895 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1896 = tosa.transpose %arg165, %1895 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %1897 = tosa.reshape %1894 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1898 = tosa.reshape %1896 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %1899 = tosa.matmul %1897, %1898 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %1900 = tosa.reshape %1899 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %1901 = tosa.reshape %arg166 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %1902 = tosa.add %1901, %1900 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %1903 = tosa.reshape %1902 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %1904 = tosa.reshape %1883 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %1905 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1906 = tosa.transpose %1904, %1905 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %1907 = tosa.reshape %1893 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1908 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1909 = tosa.transpose %1907, %1908 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1910 = tosa.reshape %1903 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %1911 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1912 = tosa.transpose %1910, %1911 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %1913 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1914 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %1915 = tosa.mul %1906, %1913 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_404 = tensor.extract_slice %1906[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_405 = tensor.extract_slice %1906[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %1916 = tensor.empty() : tensor<1x12x40x64xf32>
    %1917 = linalg.negf ins(%extracted_slice_405 : tensor<1x12x40x64xf32>) outs(%1916 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %1918 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_406 = tensor.insert_slice %1917 into %1918[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_407 = tensor.insert_slice %extracted_slice_404 into %inserted_slice_406[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %1919 = tosa.mul %inserted_slice_407, %1914 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1920 = tosa.add %1915, %1919 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1921 = tosa.mul %1909, %1913 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_408 = tensor.extract_slice %1909[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_409 = tensor.extract_slice %1909[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %1922 = tensor.empty() : tensor<1x2x40x64xf32>
    %1923 = linalg.negf ins(%extracted_slice_409 : tensor<1x2x40x64xf32>) outs(%1922 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %1924 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_410 = tensor.insert_slice %1923 into %1924[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_411 = tensor.insert_slice %extracted_slice_408 into %inserted_slice_410[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %1925 = tosa.mul %inserted_slice_411, %1914 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %1926 = tosa.add %1921, %1925 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_412 = tensor.extract_slice %1926[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_413 = tensor.extract_slice %extracted_slice_412[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1927 = tosa.reshape %extracted_slice_413 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_414 = tensor.extract_slice %1927[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_415 = tensor.extract_slice %extracted_slice_414[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1928 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1929 = tosa.add %extracted_slice_415, %1928 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1930 = tosa.identity %1929 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1931 = tosa.reshape %1930 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_416 = tensor.extract_slice %1912[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_417 = tensor.extract_slice %extracted_slice_416[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %1932 = tosa.reshape %extracted_slice_417 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_418 = tensor.extract_slice %1932[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_419 = tensor.extract_slice %extracted_slice_418[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %1933 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %1934 = tosa.add %extracted_slice_419, %1933 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1935 = tosa.identity %1934 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %1936 = tosa.reshape %1935 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_420 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_421 = tensor.extract_slice %extracted_slice_420[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_422 = tensor.extract_slice %extracted_slice_421[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_423 = arith.constant 0.000000e+00 : f32
    %splat_424 = tensor.splat %cst_423 : tensor<40x40xf32>
    %1937 = tosa.reshape %extracted_slice_422 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %1938 = tosa.add %splat_424, %1937 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %1939 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1940 = tosa.transpose %1931, %1939 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %1941 = tosa.reshape %1920 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1942 = tosa.reshape %1940 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %1943 = tosa.matmul %1941, %1942 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_425 = arith.constant 0.0883883461 : f32
    %splat_426 = tensor.splat %cst_425 : tensor<12x40x40xf32>
    %1944 = tosa.mul %1943, %splat_426 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %1945 = tosa.reshape %1938 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %1946 = tosa.add %1944, %1945 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %1947 = tosa.reduce_max %1946 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1948 = tosa.sub %1946, %1947 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1949 = math.exp %1948 : tensor<12x40x40xf32>
    %1950 = tosa.reduce_sum %1949 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %1951 = tosa.log %1950 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1952 = tosa.add %1947, %1951 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %1953 = tosa.sub %1946, %1952 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %1954 = math.exp %1953 : tensor<12x40x40xf32>
    %1955 = tosa.reshape %1952 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %1956 = tosa.reshape %1936 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %1957 = tosa.matmul %1954, %1956 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %1958 = tosa.reshape %1957 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %1959 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %1960 = tosa.transpose %1958, %1959 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %1961 = tosa.reshape %1960 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %1962 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1963 = tosa.transpose %arg167, %1962 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %1964 = tosa.reshape %1961 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_427 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1965 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1964, %1963 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_427 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1966 = tosa.reshape %1965 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1967 = tosa.add %1860, %1966 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1968 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_428 = arith.constant 2 : i32
    %1969 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1967 : tensor<1x40x1536xf32>) outs(%1968 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_428 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %1970 = tosa.reduce_sum %1969 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %1971 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1972 = tosa.reciprocal %1971 : (tensor<1xf32>) -> tensor<1xf32>
    %1973 = tosa.reshape %1972 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %1974 = tosa.mul %1973, %1970 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1975 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %1976 = tosa.add %1974, %1975 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1977 = tosa.rsqrt %1976 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %1978 = tosa.mul %1967, %1977 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %1979 = tosa.reshape %arg168 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %1980 = tosa.mul %1979, %1978 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %1981 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1982 = tosa.transpose %arg169, %1981 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1983 = tosa.reshape %1980 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_429 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1984 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1983, %1982 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_429 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1985 = tosa.reshape %1984 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1986 = tosa.sigmoid %1985 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1987 = tosa.mul %1985, %1986 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1988 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1989 = tosa.transpose %arg170, %1988 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %1990 = tosa.reshape %1980 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_430 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %1991 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1990, %1989 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_430 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %1992 = tosa.reshape %1991 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %1993 = tosa.mul %1987, %1992 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %1994 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1995 = tosa.transpose %arg171, %1994 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %1996 = tosa.reshape %1993 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_431 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %1997 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1996, %1995 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_431 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %1998 = tosa.reshape %1997 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %1999 = tosa.add %1967, %1998 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2000 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_432 = arith.constant 2 : i32
    %2001 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1999 : tensor<1x40x1536xf32>) outs(%2000 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_432 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2002 = tosa.reduce_sum %2001 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2003 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2004 = tosa.reciprocal %2003 : (tensor<1xf32>) -> tensor<1xf32>
    %2005 = tosa.reshape %2004 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2006 = tosa.mul %2005, %2002 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2007 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2008 = tosa.add %2006, %2007 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2009 = tosa.rsqrt %2008 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2010 = tosa.mul %1999, %2009 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2011 = tosa.reshape %arg172 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2012 = tosa.mul %2011, %2010 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2013 = tosa.reshape %2012 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2014 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2015 = tosa.transpose %arg173, %2014 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2016 = tosa.reshape %2013 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2017 = tosa.reshape %2015 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %2018 = tosa.matmul %2016, %2017 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %2019 = tosa.reshape %2018 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2020 = tosa.reshape %arg174 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %2021 = tosa.add %2020, %2019 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2022 = tosa.reshape %2021 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2023 = tosa.reshape %2012 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2024 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2025 = tosa.transpose %arg175, %2024 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2026 = tosa.reshape %2023 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2027 = tosa.reshape %2025 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2028 = tosa.matmul %2026, %2027 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2029 = tosa.reshape %2028 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2030 = tosa.reshape %arg176 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2031 = tosa.add %2030, %2029 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2032 = tosa.reshape %2031 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2033 = tosa.reshape %2012 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2034 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2035 = tosa.transpose %arg177, %2034 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2036 = tosa.reshape %2033 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2037 = tosa.reshape %2035 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2038 = tosa.matmul %2036, %2037 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2039 = tosa.reshape %2038 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2040 = tosa.reshape %arg178 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2041 = tosa.add %2040, %2039 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2042 = tosa.reshape %2041 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2043 = tosa.reshape %2022 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %2044 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2045 = tosa.transpose %2043, %2044 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %2046 = tosa.reshape %2032 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2047 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2048 = tosa.transpose %2046, %2047 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2049 = tosa.reshape %2042 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2050 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2051 = tosa.transpose %2049, %2050 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2052 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2053 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2054 = tosa.mul %2045, %2052 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_433 = tensor.extract_slice %2045[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_434 = tensor.extract_slice %2045[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %2055 = tensor.empty() : tensor<1x12x40x64xf32>
    %2056 = linalg.negf ins(%extracted_slice_434 : tensor<1x12x40x64xf32>) outs(%2055 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %2057 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_435 = tensor.insert_slice %2056 into %2057[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_436 = tensor.insert_slice %extracted_slice_433 into %inserted_slice_435[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %2058 = tosa.mul %inserted_slice_436, %2053 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2059 = tosa.add %2054, %2058 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2060 = tosa.mul %2048, %2052 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_437 = tensor.extract_slice %2048[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_438 = tensor.extract_slice %2048[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %2061 = tensor.empty() : tensor<1x2x40x64xf32>
    %2062 = linalg.negf ins(%extracted_slice_438 : tensor<1x2x40x64xf32>) outs(%2061 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %2063 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_439 = tensor.insert_slice %2062 into %2063[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_440 = tensor.insert_slice %extracted_slice_437 into %inserted_slice_439[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %2064 = tosa.mul %inserted_slice_440, %2053 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %2065 = tosa.add %2060, %2064 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_441 = tensor.extract_slice %2065[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_442 = tensor.extract_slice %extracted_slice_441[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2066 = tosa.reshape %extracted_slice_442 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_443 = tensor.extract_slice %2066[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_444 = tensor.extract_slice %extracted_slice_443[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2067 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2068 = tosa.add %extracted_slice_444, %2067 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2069 = tosa.identity %2068 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2070 = tosa.reshape %2069 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_445 = tensor.extract_slice %2051[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_446 = tensor.extract_slice %extracted_slice_445[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2071 = tosa.reshape %extracted_slice_446 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_447 = tensor.extract_slice %2071[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_448 = tensor.extract_slice %extracted_slice_447[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2072 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2073 = tosa.add %extracted_slice_448, %2072 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2074 = tosa.identity %2073 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2075 = tosa.reshape %2074 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_449 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_450 = tensor.extract_slice %extracted_slice_449[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_451 = tensor.extract_slice %extracted_slice_450[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_452 = arith.constant 0.000000e+00 : f32
    %splat_453 = tensor.splat %cst_452 : tensor<40x40xf32>
    %2076 = tosa.reshape %extracted_slice_451 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2077 = tosa.add %splat_453, %2076 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2078 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2079 = tosa.transpose %2070, %2078 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %2080 = tosa.reshape %2059 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2081 = tosa.reshape %2079 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %2082 = tosa.matmul %2080, %2081 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_454 = arith.constant 0.0883883461 : f32
    %splat_455 = tensor.splat %cst_454 : tensor<12x40x40xf32>
    %2083 = tosa.mul %2082, %splat_455 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %2084 = tosa.reshape %2077 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %2085 = tosa.add %2083, %2084 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %2086 = tosa.reduce_max %2085 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2087 = tosa.sub %2085, %2086 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2088 = math.exp %2087 : tensor<12x40x40xf32>
    %2089 = tosa.reduce_sum %2088 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2090 = tosa.log %2089 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2091 = tosa.add %2086, %2090 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2092 = tosa.sub %2085, %2091 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2093 = math.exp %2092 : tensor<12x40x40xf32>
    %2094 = tosa.reshape %2091 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %2095 = tosa.reshape %2075 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2096 = tosa.matmul %2093, %2095 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %2097 = tosa.reshape %2096 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2098 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2099 = tosa.transpose %2097, %2098 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %2100 = tosa.reshape %2099 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %2101 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2102 = tosa.transpose %arg179, %2101 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2103 = tosa.reshape %2100 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_456 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2104 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2103, %2102 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_456 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2105 = tosa.reshape %2104 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2106 = tosa.add %1999, %2105 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2107 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_457 = arith.constant 2 : i32
    %2108 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2106 : tensor<1x40x1536xf32>) outs(%2107 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_457 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2109 = tosa.reduce_sum %2108 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2110 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2111 = tosa.reciprocal %2110 : (tensor<1xf32>) -> tensor<1xf32>
    %2112 = tosa.reshape %2111 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2113 = tosa.mul %2112, %2109 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2114 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2115 = tosa.add %2113, %2114 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2116 = tosa.rsqrt %2115 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2117 = tosa.mul %2106, %2116 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2118 = tosa.reshape %arg180 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2119 = tosa.mul %2118, %2117 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2120 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2121 = tosa.transpose %arg181, %2120 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2122 = tosa.reshape %2119 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_458 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2123 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2122, %2121 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_458 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2124 = tosa.reshape %2123 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2125 = tosa.sigmoid %2124 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2126 = tosa.mul %2124, %2125 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2127 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2128 = tosa.transpose %arg182, %2127 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2129 = tosa.reshape %2119 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_459 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2130 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2129, %2128 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_459 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2131 = tosa.reshape %2130 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2132 = tosa.mul %2126, %2131 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2133 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2134 = tosa.transpose %arg183, %2133 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %2135 = tosa.reshape %2132 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_460 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2136 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2135, %2134 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_460 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2137 = tosa.reshape %2136 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2138 = tosa.add %2106, %2137 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2139 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_461 = arith.constant 2 : i32
    %2140 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2138 : tensor<1x40x1536xf32>) outs(%2139 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_461 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2141 = tosa.reduce_sum %2140 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2142 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2143 = tosa.reciprocal %2142 : (tensor<1xf32>) -> tensor<1xf32>
    %2144 = tosa.reshape %2143 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2145 = tosa.mul %2144, %2141 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2146 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2147 = tosa.add %2145, %2146 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2148 = tosa.rsqrt %2147 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2149 = tosa.mul %2138, %2148 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2150 = tosa.reshape %arg184 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2151 = tosa.mul %2150, %2149 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2152 = tosa.reshape %2151 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2153 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2154 = tosa.transpose %arg185, %2153 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2155 = tosa.reshape %2152 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2156 = tosa.reshape %2154 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %2157 = tosa.matmul %2155, %2156 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %2158 = tosa.reshape %2157 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2159 = tosa.reshape %arg186 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %2160 = tosa.add %2159, %2158 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2161 = tosa.reshape %2160 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2162 = tosa.reshape %2151 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2163 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2164 = tosa.transpose %arg187, %2163 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2165 = tosa.reshape %2162 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2166 = tosa.reshape %2164 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2167 = tosa.matmul %2165, %2166 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2168 = tosa.reshape %2167 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2169 = tosa.reshape %arg188 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2170 = tosa.add %2169, %2168 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2171 = tosa.reshape %2170 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2172 = tosa.reshape %2151 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2173 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2174 = tosa.transpose %arg189, %2173 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2175 = tosa.reshape %2172 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2176 = tosa.reshape %2174 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2177 = tosa.matmul %2175, %2176 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2178 = tosa.reshape %2177 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2179 = tosa.reshape %arg190 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2180 = tosa.add %2179, %2178 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2181 = tosa.reshape %2180 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2182 = tosa.reshape %2161 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %2183 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2184 = tosa.transpose %2182, %2183 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %2185 = tosa.reshape %2171 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2186 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2187 = tosa.transpose %2185, %2186 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2188 = tosa.reshape %2181 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2189 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2190 = tosa.transpose %2188, %2189 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2191 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2192 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2193 = tosa.mul %2184, %2191 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_462 = tensor.extract_slice %2184[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_463 = tensor.extract_slice %2184[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %2194 = tensor.empty() : tensor<1x12x40x64xf32>
    %2195 = linalg.negf ins(%extracted_slice_463 : tensor<1x12x40x64xf32>) outs(%2194 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %2196 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_464 = tensor.insert_slice %2195 into %2196[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_465 = tensor.insert_slice %extracted_slice_462 into %inserted_slice_464[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %2197 = tosa.mul %inserted_slice_465, %2192 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2198 = tosa.add %2193, %2197 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2199 = tosa.mul %2187, %2191 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_466 = tensor.extract_slice %2187[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_467 = tensor.extract_slice %2187[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %2200 = tensor.empty() : tensor<1x2x40x64xf32>
    %2201 = linalg.negf ins(%extracted_slice_467 : tensor<1x2x40x64xf32>) outs(%2200 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %2202 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_468 = tensor.insert_slice %2201 into %2202[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_469 = tensor.insert_slice %extracted_slice_466 into %inserted_slice_468[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %2203 = tosa.mul %inserted_slice_469, %2192 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %2204 = tosa.add %2199, %2203 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_470 = tensor.extract_slice %2204[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_471 = tensor.extract_slice %extracted_slice_470[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2205 = tosa.reshape %extracted_slice_471 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_472 = tensor.extract_slice %2205[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_473 = tensor.extract_slice %extracted_slice_472[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2206 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2207 = tosa.add %extracted_slice_473, %2206 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2208 = tosa.identity %2207 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2209 = tosa.reshape %2208 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_474 = tensor.extract_slice %2190[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_475 = tensor.extract_slice %extracted_slice_474[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2210 = tosa.reshape %extracted_slice_475 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_476 = tensor.extract_slice %2210[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_477 = tensor.extract_slice %extracted_slice_476[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2211 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2212 = tosa.add %extracted_slice_477, %2211 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2213 = tosa.identity %2212 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2214 = tosa.reshape %2213 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_478 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_479 = tensor.extract_slice %extracted_slice_478[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_480 = tensor.extract_slice %extracted_slice_479[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_481 = arith.constant 0.000000e+00 : f32
    %splat_482 = tensor.splat %cst_481 : tensor<40x40xf32>
    %2215 = tosa.reshape %extracted_slice_480 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2216 = tosa.add %splat_482, %2215 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2217 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2218 = tosa.transpose %2209, %2217 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %2219 = tosa.reshape %2198 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2220 = tosa.reshape %2218 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %2221 = tosa.matmul %2219, %2220 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_483 = arith.constant 0.0883883461 : f32
    %splat_484 = tensor.splat %cst_483 : tensor<12x40x40xf32>
    %2222 = tosa.mul %2221, %splat_484 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %2223 = tosa.reshape %2216 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %2224 = tosa.add %2222, %2223 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %2225 = tosa.reduce_max %2224 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2226 = tosa.sub %2224, %2225 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2227 = math.exp %2226 : tensor<12x40x40xf32>
    %2228 = tosa.reduce_sum %2227 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2229 = tosa.log %2228 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2230 = tosa.add %2225, %2229 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2231 = tosa.sub %2224, %2230 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2232 = math.exp %2231 : tensor<12x40x40xf32>
    %2233 = tosa.reshape %2230 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %2234 = tosa.reshape %2214 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2235 = tosa.matmul %2232, %2234 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %2236 = tosa.reshape %2235 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2237 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2238 = tosa.transpose %2236, %2237 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %2239 = tosa.reshape %2238 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %2240 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2241 = tosa.transpose %arg191, %2240 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2242 = tosa.reshape %2239 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_485 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2243 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2242, %2241 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_485 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2244 = tosa.reshape %2243 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2245 = tosa.add %2138, %2244 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2246 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_486 = arith.constant 2 : i32
    %2247 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2245 : tensor<1x40x1536xf32>) outs(%2246 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_486 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2248 = tosa.reduce_sum %2247 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2249 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2250 = tosa.reciprocal %2249 : (tensor<1xf32>) -> tensor<1xf32>
    %2251 = tosa.reshape %2250 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2252 = tosa.mul %2251, %2248 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2253 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2254 = tosa.add %2252, %2253 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2255 = tosa.rsqrt %2254 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2256 = tosa.mul %2245, %2255 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2257 = tosa.reshape %arg192 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2258 = tosa.mul %2257, %2256 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2259 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2260 = tosa.transpose %arg193, %2259 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2261 = tosa.reshape %2258 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_487 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2262 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2261, %2260 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_487 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2263 = tosa.reshape %2262 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2264 = tosa.sigmoid %2263 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2265 = tosa.mul %2263, %2264 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2266 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2267 = tosa.transpose %arg194, %2266 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2268 = tosa.reshape %2258 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_488 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2269 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2268, %2267 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_488 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2270 = tosa.reshape %2269 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2271 = tosa.mul %2265, %2270 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2272 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2273 = tosa.transpose %arg195, %2272 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %2274 = tosa.reshape %2271 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_489 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2275 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2274, %2273 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_489 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2276 = tosa.reshape %2275 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2277 = tosa.add %2245, %2276 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2278 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_490 = arith.constant 2 : i32
    %2279 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2277 : tensor<1x40x1536xf32>) outs(%2278 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_490 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2280 = tosa.reduce_sum %2279 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2281 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2282 = tosa.reciprocal %2281 : (tensor<1xf32>) -> tensor<1xf32>
    %2283 = tosa.reshape %2282 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2284 = tosa.mul %2283, %2280 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2285 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2286 = tosa.add %2284, %2285 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2287 = tosa.rsqrt %2286 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2288 = tosa.mul %2277, %2287 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2289 = tosa.reshape %arg196 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2290 = tosa.mul %2289, %2288 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2291 = tosa.reshape %2290 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2292 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2293 = tosa.transpose %arg197, %2292 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2294 = tosa.reshape %2291 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2295 = tosa.reshape %2293 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %2296 = tosa.matmul %2294, %2295 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %2297 = tosa.reshape %2296 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2298 = tosa.reshape %arg198 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %2299 = tosa.add %2298, %2297 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2300 = tosa.reshape %2299 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2301 = tosa.reshape %2290 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2302 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2303 = tosa.transpose %arg199, %2302 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2304 = tosa.reshape %2301 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2305 = tosa.reshape %2303 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2306 = tosa.matmul %2304, %2305 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2307 = tosa.reshape %2306 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2308 = tosa.reshape %arg200 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2309 = tosa.add %2308, %2307 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2310 = tosa.reshape %2309 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2311 = tosa.reshape %2290 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2312 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2313 = tosa.transpose %arg201, %2312 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2314 = tosa.reshape %2311 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2315 = tosa.reshape %2313 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2316 = tosa.matmul %2314, %2315 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2317 = tosa.reshape %2316 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2318 = tosa.reshape %arg202 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2319 = tosa.add %2318, %2317 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2320 = tosa.reshape %2319 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2321 = tosa.reshape %2300 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %2322 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2323 = tosa.transpose %2321, %2322 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %2324 = tosa.reshape %2310 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2325 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2326 = tosa.transpose %2324, %2325 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2327 = tosa.reshape %2320 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2328 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2329 = tosa.transpose %2327, %2328 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2330 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2331 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2332 = tosa.mul %2323, %2330 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_491 = tensor.extract_slice %2323[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_492 = tensor.extract_slice %2323[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %2333 = tensor.empty() : tensor<1x12x40x64xf32>
    %2334 = linalg.negf ins(%extracted_slice_492 : tensor<1x12x40x64xf32>) outs(%2333 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %2335 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_493 = tensor.insert_slice %2334 into %2335[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_494 = tensor.insert_slice %extracted_slice_491 into %inserted_slice_493[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %2336 = tosa.mul %inserted_slice_494, %2331 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2337 = tosa.add %2332, %2336 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2338 = tosa.mul %2326, %2330 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_495 = tensor.extract_slice %2326[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_496 = tensor.extract_slice %2326[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %2339 = tensor.empty() : tensor<1x2x40x64xf32>
    %2340 = linalg.negf ins(%extracted_slice_496 : tensor<1x2x40x64xf32>) outs(%2339 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %2341 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_497 = tensor.insert_slice %2340 into %2341[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_498 = tensor.insert_slice %extracted_slice_495 into %inserted_slice_497[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %2342 = tosa.mul %inserted_slice_498, %2331 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %2343 = tosa.add %2338, %2342 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_499 = tensor.extract_slice %2343[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_500 = tensor.extract_slice %extracted_slice_499[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2344 = tosa.reshape %extracted_slice_500 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_501 = tensor.extract_slice %2344[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_502 = tensor.extract_slice %extracted_slice_501[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2345 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2346 = tosa.add %extracted_slice_502, %2345 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2347 = tosa.identity %2346 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2348 = tosa.reshape %2347 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_503 = tensor.extract_slice %2329[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_504 = tensor.extract_slice %extracted_slice_503[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2349 = tosa.reshape %extracted_slice_504 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_505 = tensor.extract_slice %2349[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_506 = tensor.extract_slice %extracted_slice_505[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2350 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2351 = tosa.add %extracted_slice_506, %2350 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2352 = tosa.identity %2351 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2353 = tosa.reshape %2352 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_507 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_508 = tensor.extract_slice %extracted_slice_507[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_509 = tensor.extract_slice %extracted_slice_508[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_510 = arith.constant 0.000000e+00 : f32
    %splat_511 = tensor.splat %cst_510 : tensor<40x40xf32>
    %2354 = tosa.reshape %extracted_slice_509 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2355 = tosa.add %splat_511, %2354 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2356 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2357 = tosa.transpose %2348, %2356 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %2358 = tosa.reshape %2337 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2359 = tosa.reshape %2357 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %2360 = tosa.matmul %2358, %2359 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_512 = arith.constant 0.0883883461 : f32
    %splat_513 = tensor.splat %cst_512 : tensor<12x40x40xf32>
    %2361 = tosa.mul %2360, %splat_513 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %2362 = tosa.reshape %2355 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %2363 = tosa.add %2361, %2362 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %2364 = tosa.reduce_max %2363 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2365 = tosa.sub %2363, %2364 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2366 = math.exp %2365 : tensor<12x40x40xf32>
    %2367 = tosa.reduce_sum %2366 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2368 = tosa.log %2367 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2369 = tosa.add %2364, %2368 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2370 = tosa.sub %2363, %2369 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2371 = math.exp %2370 : tensor<12x40x40xf32>
    %2372 = tosa.reshape %2369 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %2373 = tosa.reshape %2353 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2374 = tosa.matmul %2371, %2373 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %2375 = tosa.reshape %2374 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2376 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2377 = tosa.transpose %2375, %2376 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %2378 = tosa.reshape %2377 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %2379 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2380 = tosa.transpose %arg203, %2379 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2381 = tosa.reshape %2378 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_514 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2382 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2381, %2380 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_514 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2383 = tosa.reshape %2382 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2384 = tosa.add %2277, %2383 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2385 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_515 = arith.constant 2 : i32
    %2386 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2384 : tensor<1x40x1536xf32>) outs(%2385 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_515 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2387 = tosa.reduce_sum %2386 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2388 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2389 = tosa.reciprocal %2388 : (tensor<1xf32>) -> tensor<1xf32>
    %2390 = tosa.reshape %2389 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2391 = tosa.mul %2390, %2387 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2392 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2393 = tosa.add %2391, %2392 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2394 = tosa.rsqrt %2393 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2395 = tosa.mul %2384, %2394 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2396 = tosa.reshape %arg204 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2397 = tosa.mul %2396, %2395 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2398 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2399 = tosa.transpose %arg205, %2398 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2400 = tosa.reshape %2397 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_516 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2401 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2400, %2399 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_516 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2402 = tosa.reshape %2401 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2403 = tosa.sigmoid %2402 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2404 = tosa.mul %2402, %2403 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2405 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2406 = tosa.transpose %arg206, %2405 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2407 = tosa.reshape %2397 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_517 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2408 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2407, %2406 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_517 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2409 = tosa.reshape %2408 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2410 = tosa.mul %2404, %2409 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2411 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2412 = tosa.transpose %arg207, %2411 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %2413 = tosa.reshape %2410 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_518 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2414 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2413, %2412 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_518 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2415 = tosa.reshape %2414 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2416 = tosa.add %2384, %2415 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2417 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_519 = arith.constant 2 : i32
    %2418 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2416 : tensor<1x40x1536xf32>) outs(%2417 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_519 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2419 = tosa.reduce_sum %2418 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2420 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2421 = tosa.reciprocal %2420 : (tensor<1xf32>) -> tensor<1xf32>
    %2422 = tosa.reshape %2421 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2423 = tosa.mul %2422, %2419 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2424 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2425 = tosa.add %2423, %2424 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2426 = tosa.rsqrt %2425 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2427 = tosa.mul %2416, %2426 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2428 = tosa.reshape %arg208 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2429 = tosa.mul %2428, %2427 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2430 = tosa.reshape %2429 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2431 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2432 = tosa.transpose %arg209, %2431 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2433 = tosa.reshape %2430 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2434 = tosa.reshape %2432 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %2435 = tosa.matmul %2433, %2434 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %2436 = tosa.reshape %2435 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2437 = tosa.reshape %arg210 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %2438 = tosa.add %2437, %2436 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2439 = tosa.reshape %2438 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2440 = tosa.reshape %2429 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2441 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2442 = tosa.transpose %arg211, %2441 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2443 = tosa.reshape %2440 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2444 = tosa.reshape %2442 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2445 = tosa.matmul %2443, %2444 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2446 = tosa.reshape %2445 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2447 = tosa.reshape %arg212 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2448 = tosa.add %2447, %2446 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2449 = tosa.reshape %2448 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2450 = tosa.reshape %2429 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2451 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2452 = tosa.transpose %arg213, %2451 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2453 = tosa.reshape %2450 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2454 = tosa.reshape %2452 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2455 = tosa.matmul %2453, %2454 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2456 = tosa.reshape %2455 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2457 = tosa.reshape %arg214 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2458 = tosa.add %2457, %2456 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2459 = tosa.reshape %2458 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2460 = tosa.reshape %2439 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %2461 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2462 = tosa.transpose %2460, %2461 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %2463 = tosa.reshape %2449 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2464 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2465 = tosa.transpose %2463, %2464 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2466 = tosa.reshape %2459 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2467 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2468 = tosa.transpose %2466, %2467 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2469 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2470 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2471 = tosa.mul %2462, %2469 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_520 = tensor.extract_slice %2462[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_521 = tensor.extract_slice %2462[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %2472 = tensor.empty() : tensor<1x12x40x64xf32>
    %2473 = linalg.negf ins(%extracted_slice_521 : tensor<1x12x40x64xf32>) outs(%2472 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %2474 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_522 = tensor.insert_slice %2473 into %2474[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_523 = tensor.insert_slice %extracted_slice_520 into %inserted_slice_522[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %2475 = tosa.mul %inserted_slice_523, %2470 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2476 = tosa.add %2471, %2475 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2477 = tosa.mul %2465, %2469 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_524 = tensor.extract_slice %2465[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_525 = tensor.extract_slice %2465[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %2478 = tensor.empty() : tensor<1x2x40x64xf32>
    %2479 = linalg.negf ins(%extracted_slice_525 : tensor<1x2x40x64xf32>) outs(%2478 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %2480 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_526 = tensor.insert_slice %2479 into %2480[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_527 = tensor.insert_slice %extracted_slice_524 into %inserted_slice_526[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %2481 = tosa.mul %inserted_slice_527, %2470 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %2482 = tosa.add %2477, %2481 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_528 = tensor.extract_slice %2482[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_529 = tensor.extract_slice %extracted_slice_528[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2483 = tosa.reshape %extracted_slice_529 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_530 = tensor.extract_slice %2483[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_531 = tensor.extract_slice %extracted_slice_530[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2484 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2485 = tosa.add %extracted_slice_531, %2484 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2486 = tosa.identity %2485 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2487 = tosa.reshape %2486 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_532 = tensor.extract_slice %2468[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_533 = tensor.extract_slice %extracted_slice_532[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2488 = tosa.reshape %extracted_slice_533 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_534 = tensor.extract_slice %2488[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_535 = tensor.extract_slice %extracted_slice_534[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2489 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2490 = tosa.add %extracted_slice_535, %2489 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2491 = tosa.identity %2490 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2492 = tosa.reshape %2491 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_536 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_537 = tensor.extract_slice %extracted_slice_536[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_538 = tensor.extract_slice %extracted_slice_537[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_539 = arith.constant 0.000000e+00 : f32
    %splat_540 = tensor.splat %cst_539 : tensor<40x40xf32>
    %2493 = tosa.reshape %extracted_slice_538 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2494 = tosa.add %splat_540, %2493 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2495 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2496 = tosa.transpose %2487, %2495 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %2497 = tosa.reshape %2476 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2498 = tosa.reshape %2496 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %2499 = tosa.matmul %2497, %2498 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_541 = arith.constant 0.0883883461 : f32
    %splat_542 = tensor.splat %cst_541 : tensor<12x40x40xf32>
    %2500 = tosa.mul %2499, %splat_542 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %2501 = tosa.reshape %2494 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %2502 = tosa.add %2500, %2501 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %2503 = tosa.reduce_max %2502 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2504 = tosa.sub %2502, %2503 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2505 = math.exp %2504 : tensor<12x40x40xf32>
    %2506 = tosa.reduce_sum %2505 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2507 = tosa.log %2506 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2508 = tosa.add %2503, %2507 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2509 = tosa.sub %2502, %2508 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2510 = math.exp %2509 : tensor<12x40x40xf32>
    %2511 = tosa.reshape %2508 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %2512 = tosa.reshape %2492 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2513 = tosa.matmul %2510, %2512 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %2514 = tosa.reshape %2513 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2515 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2516 = tosa.transpose %2514, %2515 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %2517 = tosa.reshape %2516 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %2518 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2519 = tosa.transpose %arg215, %2518 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2520 = tosa.reshape %2517 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_543 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2521 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2520, %2519 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_543 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2522 = tosa.reshape %2521 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2523 = tosa.add %2416, %2522 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2524 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_544 = arith.constant 2 : i32
    %2525 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2523 : tensor<1x40x1536xf32>) outs(%2524 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_544 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2526 = tosa.reduce_sum %2525 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2527 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2528 = tosa.reciprocal %2527 : (tensor<1xf32>) -> tensor<1xf32>
    %2529 = tosa.reshape %2528 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2530 = tosa.mul %2529, %2526 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2531 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2532 = tosa.add %2530, %2531 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2533 = tosa.rsqrt %2532 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2534 = tosa.mul %2523, %2533 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2535 = tosa.reshape %arg216 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2536 = tosa.mul %2535, %2534 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2537 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2538 = tosa.transpose %arg217, %2537 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2539 = tosa.reshape %2536 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_545 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2540 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2539, %2538 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_545 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2541 = tosa.reshape %2540 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2542 = tosa.sigmoid %2541 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2543 = tosa.mul %2541, %2542 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2544 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2545 = tosa.transpose %arg218, %2544 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2546 = tosa.reshape %2536 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_546 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2547 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2546, %2545 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_546 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2548 = tosa.reshape %2547 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2549 = tosa.mul %2543, %2548 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2550 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2551 = tosa.transpose %arg219, %2550 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %2552 = tosa.reshape %2549 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_547 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2553 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2552, %2551 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_547 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2554 = tosa.reshape %2553 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2555 = tosa.add %2523, %2554 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2556 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_548 = arith.constant 2 : i32
    %2557 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2555 : tensor<1x40x1536xf32>) outs(%2556 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_548 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2558 = tosa.reduce_sum %2557 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2559 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2560 = tosa.reciprocal %2559 : (tensor<1xf32>) -> tensor<1xf32>
    %2561 = tosa.reshape %2560 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2562 = tosa.mul %2561, %2558 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2563 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2564 = tosa.add %2562, %2563 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2565 = tosa.rsqrt %2564 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2566 = tosa.mul %2555, %2565 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2567 = tosa.reshape %arg220 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2568 = tosa.mul %2567, %2566 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2569 = tosa.reshape %2568 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2570 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2571 = tosa.transpose %arg221, %2570 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2572 = tosa.reshape %2569 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2573 = tosa.reshape %2571 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %2574 = tosa.matmul %2572, %2573 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %2575 = tosa.reshape %2574 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2576 = tosa.reshape %arg222 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %2577 = tosa.add %2576, %2575 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2578 = tosa.reshape %2577 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2579 = tosa.reshape %2568 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2580 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2581 = tosa.transpose %arg223, %2580 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2582 = tosa.reshape %2579 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2583 = tosa.reshape %2581 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2584 = tosa.matmul %2582, %2583 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2585 = tosa.reshape %2584 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2586 = tosa.reshape %arg224 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2587 = tosa.add %2586, %2585 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2588 = tosa.reshape %2587 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2589 = tosa.reshape %2568 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2590 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2591 = tosa.transpose %arg225, %2590 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2592 = tosa.reshape %2589 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2593 = tosa.reshape %2591 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2594 = tosa.matmul %2592, %2593 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2595 = tosa.reshape %2594 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2596 = tosa.reshape %arg226 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2597 = tosa.add %2596, %2595 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2598 = tosa.reshape %2597 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2599 = tosa.reshape %2578 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %2600 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2601 = tosa.transpose %2599, %2600 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %2602 = tosa.reshape %2588 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2603 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2604 = tosa.transpose %2602, %2603 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2605 = tosa.reshape %2598 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2606 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2607 = tosa.transpose %2605, %2606 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2608 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2609 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2610 = tosa.mul %2601, %2608 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_549 = tensor.extract_slice %2601[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_550 = tensor.extract_slice %2601[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %2611 = tensor.empty() : tensor<1x12x40x64xf32>
    %2612 = linalg.negf ins(%extracted_slice_550 : tensor<1x12x40x64xf32>) outs(%2611 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %2613 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_551 = tensor.insert_slice %2612 into %2613[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_552 = tensor.insert_slice %extracted_slice_549 into %inserted_slice_551[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %2614 = tosa.mul %inserted_slice_552, %2609 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2615 = tosa.add %2610, %2614 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2616 = tosa.mul %2604, %2608 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_553 = tensor.extract_slice %2604[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_554 = tensor.extract_slice %2604[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %2617 = tensor.empty() : tensor<1x2x40x64xf32>
    %2618 = linalg.negf ins(%extracted_slice_554 : tensor<1x2x40x64xf32>) outs(%2617 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %2619 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_555 = tensor.insert_slice %2618 into %2619[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_556 = tensor.insert_slice %extracted_slice_553 into %inserted_slice_555[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %2620 = tosa.mul %inserted_slice_556, %2609 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %2621 = tosa.add %2616, %2620 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_557 = tensor.extract_slice %2621[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_558 = tensor.extract_slice %extracted_slice_557[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2622 = tosa.reshape %extracted_slice_558 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_559 = tensor.extract_slice %2622[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_560 = tensor.extract_slice %extracted_slice_559[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2623 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2624 = tosa.add %extracted_slice_560, %2623 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2625 = tosa.identity %2624 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2626 = tosa.reshape %2625 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_561 = tensor.extract_slice %2607[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_562 = tensor.extract_slice %extracted_slice_561[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2627 = tosa.reshape %extracted_slice_562 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_563 = tensor.extract_slice %2627[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_564 = tensor.extract_slice %extracted_slice_563[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2628 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2629 = tosa.add %extracted_slice_564, %2628 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2630 = tosa.identity %2629 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2631 = tosa.reshape %2630 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_565 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_566 = tensor.extract_slice %extracted_slice_565[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_567 = tensor.extract_slice %extracted_slice_566[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_568 = arith.constant 0.000000e+00 : f32
    %splat_569 = tensor.splat %cst_568 : tensor<40x40xf32>
    %2632 = tosa.reshape %extracted_slice_567 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2633 = tosa.add %splat_569, %2632 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2634 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2635 = tosa.transpose %2626, %2634 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %2636 = tosa.reshape %2615 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2637 = tosa.reshape %2635 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %2638 = tosa.matmul %2636, %2637 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_570 = arith.constant 0.0883883461 : f32
    %splat_571 = tensor.splat %cst_570 : tensor<12x40x40xf32>
    %2639 = tosa.mul %2638, %splat_571 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %2640 = tosa.reshape %2633 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %2641 = tosa.add %2639, %2640 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %2642 = tosa.reduce_max %2641 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2643 = tosa.sub %2641, %2642 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2644 = math.exp %2643 : tensor<12x40x40xf32>
    %2645 = tosa.reduce_sum %2644 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2646 = tosa.log %2645 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2647 = tosa.add %2642, %2646 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2648 = tosa.sub %2641, %2647 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2649 = math.exp %2648 : tensor<12x40x40xf32>
    %2650 = tosa.reshape %2647 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %2651 = tosa.reshape %2631 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2652 = tosa.matmul %2649, %2651 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %2653 = tosa.reshape %2652 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2654 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2655 = tosa.transpose %2653, %2654 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %2656 = tosa.reshape %2655 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %2657 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2658 = tosa.transpose %arg227, %2657 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2659 = tosa.reshape %2656 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_572 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2660 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2659, %2658 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_572 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2661 = tosa.reshape %2660 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2662 = tosa.add %2555, %2661 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2663 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_573 = arith.constant 2 : i32
    %2664 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2662 : tensor<1x40x1536xf32>) outs(%2663 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_573 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2665 = tosa.reduce_sum %2664 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2666 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2667 = tosa.reciprocal %2666 : (tensor<1xf32>) -> tensor<1xf32>
    %2668 = tosa.reshape %2667 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2669 = tosa.mul %2668, %2665 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2670 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2671 = tosa.add %2669, %2670 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2672 = tosa.rsqrt %2671 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2673 = tosa.mul %2662, %2672 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2674 = tosa.reshape %arg228 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2675 = tosa.mul %2674, %2673 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2676 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2677 = tosa.transpose %arg229, %2676 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2678 = tosa.reshape %2675 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_574 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2679 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2678, %2677 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_574 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2680 = tosa.reshape %2679 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2681 = tosa.sigmoid %2680 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2682 = tosa.mul %2680, %2681 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2683 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2684 = tosa.transpose %arg230, %2683 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2685 = tosa.reshape %2675 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_575 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2686 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2685, %2684 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_575 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2687 = tosa.reshape %2686 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2688 = tosa.mul %2682, %2687 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2689 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2690 = tosa.transpose %arg231, %2689 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %2691 = tosa.reshape %2688 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_576 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2692 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2691, %2690 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_576 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2693 = tosa.reshape %2692 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2694 = tosa.add %2662, %2693 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2695 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_577 = arith.constant 2 : i32
    %2696 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2694 : tensor<1x40x1536xf32>) outs(%2695 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_577 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2697 = tosa.reduce_sum %2696 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2698 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2699 = tosa.reciprocal %2698 : (tensor<1xf32>) -> tensor<1xf32>
    %2700 = tosa.reshape %2699 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2701 = tosa.mul %2700, %2697 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2702 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2703 = tosa.add %2701, %2702 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2704 = tosa.rsqrt %2703 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2705 = tosa.mul %2694, %2704 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2706 = tosa.reshape %arg232 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2707 = tosa.mul %2706, %2705 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2708 = tosa.reshape %2707 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2709 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2710 = tosa.transpose %arg233, %2709 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2711 = tosa.reshape %2708 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2712 = tosa.reshape %2710 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %2713 = tosa.matmul %2711, %2712 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %2714 = tosa.reshape %2713 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2715 = tosa.reshape %arg234 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %2716 = tosa.add %2715, %2714 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2717 = tosa.reshape %2716 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2718 = tosa.reshape %2707 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2719 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2720 = tosa.transpose %arg235, %2719 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2721 = tosa.reshape %2718 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2722 = tosa.reshape %2720 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2723 = tosa.matmul %2721, %2722 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2724 = tosa.reshape %2723 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2725 = tosa.reshape %arg236 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2726 = tosa.add %2725, %2724 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2727 = tosa.reshape %2726 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2728 = tosa.reshape %2707 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2729 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2730 = tosa.transpose %arg237, %2729 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2731 = tosa.reshape %2728 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2732 = tosa.reshape %2730 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2733 = tosa.matmul %2731, %2732 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2734 = tosa.reshape %2733 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2735 = tosa.reshape %arg238 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2736 = tosa.add %2735, %2734 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2737 = tosa.reshape %2736 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2738 = tosa.reshape %2717 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %2739 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2740 = tosa.transpose %2738, %2739 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %2741 = tosa.reshape %2727 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2742 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2743 = tosa.transpose %2741, %2742 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2744 = tosa.reshape %2737 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2745 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2746 = tosa.transpose %2744, %2745 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2747 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2748 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2749 = tosa.mul %2740, %2747 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_578 = tensor.extract_slice %2740[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_579 = tensor.extract_slice %2740[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %2750 = tensor.empty() : tensor<1x12x40x64xf32>
    %2751 = linalg.negf ins(%extracted_slice_579 : tensor<1x12x40x64xf32>) outs(%2750 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %2752 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_580 = tensor.insert_slice %2751 into %2752[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_581 = tensor.insert_slice %extracted_slice_578 into %inserted_slice_580[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %2753 = tosa.mul %inserted_slice_581, %2748 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2754 = tosa.add %2749, %2753 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2755 = tosa.mul %2743, %2747 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_582 = tensor.extract_slice %2743[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_583 = tensor.extract_slice %2743[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %2756 = tensor.empty() : tensor<1x2x40x64xf32>
    %2757 = linalg.negf ins(%extracted_slice_583 : tensor<1x2x40x64xf32>) outs(%2756 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %2758 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_584 = tensor.insert_slice %2757 into %2758[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_585 = tensor.insert_slice %extracted_slice_582 into %inserted_slice_584[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %2759 = tosa.mul %inserted_slice_585, %2748 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %2760 = tosa.add %2755, %2759 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_586 = tensor.extract_slice %2760[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_587 = tensor.extract_slice %extracted_slice_586[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2761 = tosa.reshape %extracted_slice_587 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_588 = tensor.extract_slice %2761[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_589 = tensor.extract_slice %extracted_slice_588[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2762 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2763 = tosa.add %extracted_slice_589, %2762 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2764 = tosa.identity %2763 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2765 = tosa.reshape %2764 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_590 = tensor.extract_slice %2746[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_591 = tensor.extract_slice %extracted_slice_590[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2766 = tosa.reshape %extracted_slice_591 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_592 = tensor.extract_slice %2766[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_593 = tensor.extract_slice %extracted_slice_592[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2767 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2768 = tosa.add %extracted_slice_593, %2767 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2769 = tosa.identity %2768 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2770 = tosa.reshape %2769 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_594 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_595 = tensor.extract_slice %extracted_slice_594[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_596 = tensor.extract_slice %extracted_slice_595[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_597 = arith.constant 0.000000e+00 : f32
    %splat_598 = tensor.splat %cst_597 : tensor<40x40xf32>
    %2771 = tosa.reshape %extracted_slice_596 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2772 = tosa.add %splat_598, %2771 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2773 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2774 = tosa.transpose %2765, %2773 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %2775 = tosa.reshape %2754 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2776 = tosa.reshape %2774 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %2777 = tosa.matmul %2775, %2776 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_599 = arith.constant 0.0883883461 : f32
    %splat_600 = tensor.splat %cst_599 : tensor<12x40x40xf32>
    %2778 = tosa.mul %2777, %splat_600 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %2779 = tosa.reshape %2772 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %2780 = tosa.add %2778, %2779 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %2781 = tosa.reduce_max %2780 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2782 = tosa.sub %2780, %2781 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2783 = math.exp %2782 : tensor<12x40x40xf32>
    %2784 = tosa.reduce_sum %2783 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2785 = tosa.log %2784 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2786 = tosa.add %2781, %2785 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2787 = tosa.sub %2780, %2786 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2788 = math.exp %2787 : tensor<12x40x40xf32>
    %2789 = tosa.reshape %2786 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %2790 = tosa.reshape %2770 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2791 = tosa.matmul %2788, %2790 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %2792 = tosa.reshape %2791 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2793 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2794 = tosa.transpose %2792, %2793 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %2795 = tosa.reshape %2794 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %2796 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2797 = tosa.transpose %arg239, %2796 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2798 = tosa.reshape %2795 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_601 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2799 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2798, %2797 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_601 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2800 = tosa.reshape %2799 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2801 = tosa.add %2694, %2800 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2802 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_602 = arith.constant 2 : i32
    %2803 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2801 : tensor<1x40x1536xf32>) outs(%2802 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_602 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2804 = tosa.reduce_sum %2803 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2805 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2806 = tosa.reciprocal %2805 : (tensor<1xf32>) -> tensor<1xf32>
    %2807 = tosa.reshape %2806 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2808 = tosa.mul %2807, %2804 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2809 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2810 = tosa.add %2808, %2809 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2811 = tosa.rsqrt %2810 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2812 = tosa.mul %2801, %2811 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2813 = tosa.reshape %arg240 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2814 = tosa.mul %2813, %2812 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2815 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2816 = tosa.transpose %arg241, %2815 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2817 = tosa.reshape %2814 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_603 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2818 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2817, %2816 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_603 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2819 = tosa.reshape %2818 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2820 = tosa.sigmoid %2819 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2821 = tosa.mul %2819, %2820 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2822 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2823 = tosa.transpose %arg242, %2822 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2824 = tosa.reshape %2814 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_604 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2825 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2824, %2823 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_604 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2826 = tosa.reshape %2825 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2827 = tosa.mul %2821, %2826 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2828 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2829 = tosa.transpose %arg243, %2828 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %2830 = tosa.reshape %2827 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_605 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2831 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2830, %2829 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_605 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2832 = tosa.reshape %2831 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2833 = tosa.add %2801, %2832 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2834 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_606 = arith.constant 2 : i32
    %2835 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2833 : tensor<1x40x1536xf32>) outs(%2834 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_606 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2836 = tosa.reduce_sum %2835 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2837 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2838 = tosa.reciprocal %2837 : (tensor<1xf32>) -> tensor<1xf32>
    %2839 = tosa.reshape %2838 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2840 = tosa.mul %2839, %2836 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2841 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2842 = tosa.add %2840, %2841 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2843 = tosa.rsqrt %2842 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2844 = tosa.mul %2833, %2843 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2845 = tosa.reshape %arg244 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2846 = tosa.mul %2845, %2844 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2847 = tosa.reshape %2846 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2848 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2849 = tosa.transpose %arg245, %2848 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2850 = tosa.reshape %2847 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2851 = tosa.reshape %2849 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %2852 = tosa.matmul %2850, %2851 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %2853 = tosa.reshape %2852 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2854 = tosa.reshape %arg246 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %2855 = tosa.add %2854, %2853 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2856 = tosa.reshape %2855 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2857 = tosa.reshape %2846 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2858 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2859 = tosa.transpose %arg247, %2858 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2860 = tosa.reshape %2857 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2861 = tosa.reshape %2859 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2862 = tosa.matmul %2860, %2861 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2863 = tosa.reshape %2862 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2864 = tosa.reshape %arg248 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2865 = tosa.add %2864, %2863 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2866 = tosa.reshape %2865 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2867 = tosa.reshape %2846 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2868 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2869 = tosa.transpose %arg249, %2868 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2870 = tosa.reshape %2867 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2871 = tosa.reshape %2869 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %2872 = tosa.matmul %2870, %2871 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %2873 = tosa.reshape %2872 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %2874 = tosa.reshape %arg250 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %2875 = tosa.add %2874, %2873 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %2876 = tosa.reshape %2875 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %2877 = tosa.reshape %2856 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %2878 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2879 = tosa.transpose %2877, %2878 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %2880 = tosa.reshape %2866 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2881 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2882 = tosa.transpose %2880, %2881 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2883 = tosa.reshape %2876 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %2884 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2885 = tosa.transpose %2883, %2884 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %2886 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2887 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %2888 = tosa.mul %2879, %2886 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_607 = tensor.extract_slice %2879[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_608 = tensor.extract_slice %2879[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %2889 = tensor.empty() : tensor<1x12x40x64xf32>
    %2890 = linalg.negf ins(%extracted_slice_608 : tensor<1x12x40x64xf32>) outs(%2889 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %2891 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_609 = tensor.insert_slice %2890 into %2891[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_610 = tensor.insert_slice %extracted_slice_607 into %inserted_slice_609[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %2892 = tosa.mul %inserted_slice_610, %2887 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2893 = tosa.add %2888, %2892 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2894 = tosa.mul %2882, %2886 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_611 = tensor.extract_slice %2882[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_612 = tensor.extract_slice %2882[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %2895 = tensor.empty() : tensor<1x2x40x64xf32>
    %2896 = linalg.negf ins(%extracted_slice_612 : tensor<1x2x40x64xf32>) outs(%2895 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %2897 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_613 = tensor.insert_slice %2896 into %2897[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_614 = tensor.insert_slice %extracted_slice_611 into %inserted_slice_613[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %2898 = tosa.mul %inserted_slice_614, %2887 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %2899 = tosa.add %2894, %2898 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_615 = tensor.extract_slice %2899[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_616 = tensor.extract_slice %extracted_slice_615[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2900 = tosa.reshape %extracted_slice_616 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_617 = tensor.extract_slice %2900[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_618 = tensor.extract_slice %extracted_slice_617[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2901 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2902 = tosa.add %extracted_slice_618, %2901 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2903 = tosa.identity %2902 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2904 = tosa.reshape %2903 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_619 = tensor.extract_slice %2885[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_620 = tensor.extract_slice %extracted_slice_619[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %2905 = tosa.reshape %extracted_slice_620 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_621 = tensor.extract_slice %2905[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_622 = tensor.extract_slice %extracted_slice_621[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %2906 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %2907 = tosa.add %extracted_slice_622, %2906 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2908 = tosa.identity %2907 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %2909 = tosa.reshape %2908 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_623 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_624 = tensor.extract_slice %extracted_slice_623[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_625 = tensor.extract_slice %extracted_slice_624[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_626 = arith.constant 0.000000e+00 : f32
    %splat_627 = tensor.splat %cst_626 : tensor<40x40xf32>
    %2910 = tosa.reshape %extracted_slice_625 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %2911 = tosa.add %splat_627, %2910 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %2912 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2913 = tosa.transpose %2904, %2912 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %2914 = tosa.reshape %2893 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2915 = tosa.reshape %2913 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %2916 = tosa.matmul %2914, %2915 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_628 = arith.constant 0.0883883461 : f32
    %splat_629 = tensor.splat %cst_628 : tensor<12x40x40xf32>
    %2917 = tosa.mul %2916, %splat_629 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %2918 = tosa.reshape %2911 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %2919 = tosa.add %2917, %2918 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %2920 = tosa.reduce_max %2919 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2921 = tosa.sub %2919, %2920 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2922 = math.exp %2921 : tensor<12x40x40xf32>
    %2923 = tosa.reduce_sum %2922 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %2924 = tosa.log %2923 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2925 = tosa.add %2920, %2924 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %2926 = tosa.sub %2919, %2925 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %2927 = math.exp %2926 : tensor<12x40x40xf32>
    %2928 = tosa.reshape %2925 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %2929 = tosa.reshape %2909 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %2930 = tosa.matmul %2927, %2929 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %2931 = tosa.reshape %2930 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %2932 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2933 = tosa.transpose %2931, %2932 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %2934 = tosa.reshape %2933 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %2935 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2936 = tosa.transpose %arg251, %2935 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2937 = tosa.reshape %2934 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_630 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2938 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2937, %2936 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_630 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2939 = tosa.reshape %2938 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2940 = tosa.add %2833, %2939 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2941 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_631 = arith.constant 2 : i32
    %2942 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2940 : tensor<1x40x1536xf32>) outs(%2941 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_631 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2943 = tosa.reduce_sum %2942 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2944 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2945 = tosa.reciprocal %2944 : (tensor<1xf32>) -> tensor<1xf32>
    %2946 = tosa.reshape %2945 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2947 = tosa.mul %2946, %2943 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2948 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2949 = tosa.add %2947, %2948 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2950 = tosa.rsqrt %2949 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2951 = tosa.mul %2940, %2950 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2952 = tosa.reshape %arg252 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2953 = tosa.mul %2952, %2951 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2954 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2955 = tosa.transpose %arg253, %2954 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2956 = tosa.reshape %2953 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_632 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2957 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2956, %2955 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_632 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2958 = tosa.reshape %2957 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2959 = tosa.sigmoid %2958 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2960 = tosa.mul %2958, %2959 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2961 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2962 = tosa.transpose %arg254, %2961 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %2963 = tosa.reshape %2953 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_633 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %2964 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2963, %2962 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_633 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %2965 = tosa.reshape %2964 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %2966 = tosa.mul %2960, %2965 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %2967 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2968 = tosa.transpose %arg255, %2967 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %2969 = tosa.reshape %2966 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_634 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %2970 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2969, %2968 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_634 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2971 = tosa.reshape %2970 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2972 = tosa.add %2940, %2971 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2973 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_635 = arith.constant 2 : i32
    %2974 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2972 : tensor<1x40x1536xf32>) outs(%2973 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_635 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %2975 = tosa.reduce_sum %2974 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %2976 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2977 = tosa.reciprocal %2976 : (tensor<1xf32>) -> tensor<1xf32>
    %2978 = tosa.reshape %2977 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %2979 = tosa.mul %2978, %2975 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2980 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %2981 = tosa.add %2979, %2980 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2982 = tosa.rsqrt %2981 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %2983 = tosa.mul %2972, %2982 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %2984 = tosa.reshape %arg256 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %2985 = tosa.mul %2984, %2983 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %2986 = tosa.reshape %2985 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2987 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2988 = tosa.transpose %arg257, %2987 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %2989 = tosa.reshape %2986 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2990 = tosa.reshape %2988 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %2991 = tosa.matmul %2989, %2990 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %2992 = tosa.reshape %2991 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2993 = tosa.reshape %arg258 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %2994 = tosa.add %2993, %2992 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %2995 = tosa.reshape %2994 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %2996 = tosa.reshape %2985 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %2997 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2998 = tosa.transpose %arg259, %2997 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %2999 = tosa.reshape %2996 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3000 = tosa.reshape %2998 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3001 = tosa.matmul %2999, %3000 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3002 = tosa.reshape %3001 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3003 = tosa.reshape %arg260 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3004 = tosa.add %3003, %3002 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3005 = tosa.reshape %3004 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3006 = tosa.reshape %2985 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3007 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3008 = tosa.transpose %arg261, %3007 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3009 = tosa.reshape %3006 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3010 = tosa.reshape %3008 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3011 = tosa.matmul %3009, %3010 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3012 = tosa.reshape %3011 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3013 = tosa.reshape %arg262 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3014 = tosa.add %3013, %3012 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3015 = tosa.reshape %3014 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3016 = tosa.reshape %2995 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %3017 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3018 = tosa.transpose %3016, %3017 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %3019 = tosa.reshape %3005 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3020 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3021 = tosa.transpose %3019, %3020 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3022 = tosa.reshape %3015 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3023 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3024 = tosa.transpose %3022, %3023 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3025 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3026 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3027 = tosa.mul %3018, %3025 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_636 = tensor.extract_slice %3018[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_637 = tensor.extract_slice %3018[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %3028 = tensor.empty() : tensor<1x12x40x64xf32>
    %3029 = linalg.negf ins(%extracted_slice_637 : tensor<1x12x40x64xf32>) outs(%3028 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %3030 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_638 = tensor.insert_slice %3029 into %3030[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_639 = tensor.insert_slice %extracted_slice_636 into %inserted_slice_638[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %3031 = tosa.mul %inserted_slice_639, %3026 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3032 = tosa.add %3027, %3031 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3033 = tosa.mul %3021, %3025 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_640 = tensor.extract_slice %3021[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_641 = tensor.extract_slice %3021[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %3034 = tensor.empty() : tensor<1x2x40x64xf32>
    %3035 = linalg.negf ins(%extracted_slice_641 : tensor<1x2x40x64xf32>) outs(%3034 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %3036 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_642 = tensor.insert_slice %3035 into %3036[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_643 = tensor.insert_slice %extracted_slice_640 into %inserted_slice_642[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %3037 = tosa.mul %inserted_slice_643, %3026 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %3038 = tosa.add %3033, %3037 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_644 = tensor.extract_slice %3038[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_645 = tensor.extract_slice %extracted_slice_644[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3039 = tosa.reshape %extracted_slice_645 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_646 = tensor.extract_slice %3039[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_647 = tensor.extract_slice %extracted_slice_646[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3040 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3041 = tosa.add %extracted_slice_647, %3040 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3042 = tosa.identity %3041 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3043 = tosa.reshape %3042 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_648 = tensor.extract_slice %3024[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_649 = tensor.extract_slice %extracted_slice_648[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3044 = tosa.reshape %extracted_slice_649 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_650 = tensor.extract_slice %3044[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_651 = tensor.extract_slice %extracted_slice_650[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3045 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3046 = tosa.add %extracted_slice_651, %3045 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3047 = tosa.identity %3046 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3048 = tosa.reshape %3047 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_652 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_653 = tensor.extract_slice %extracted_slice_652[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_654 = tensor.extract_slice %extracted_slice_653[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_655 = arith.constant 0.000000e+00 : f32
    %splat_656 = tensor.splat %cst_655 : tensor<40x40xf32>
    %3049 = tosa.reshape %extracted_slice_654 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3050 = tosa.add %splat_656, %3049 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3051 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3052 = tosa.transpose %3043, %3051 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %3053 = tosa.reshape %3032 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3054 = tosa.reshape %3052 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %3055 = tosa.matmul %3053, %3054 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_657 = arith.constant 0.0883883461 : f32
    %splat_658 = tensor.splat %cst_657 : tensor<12x40x40xf32>
    %3056 = tosa.mul %3055, %splat_658 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %3057 = tosa.reshape %3050 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %3058 = tosa.add %3056, %3057 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %3059 = tosa.reduce_max %3058 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3060 = tosa.sub %3058, %3059 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3061 = math.exp %3060 : tensor<12x40x40xf32>
    %3062 = tosa.reduce_sum %3061 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3063 = tosa.log %3062 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3064 = tosa.add %3059, %3063 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3065 = tosa.sub %3058, %3064 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3066 = math.exp %3065 : tensor<12x40x40xf32>
    %3067 = tosa.reshape %3064 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %3068 = tosa.reshape %3048 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3069 = tosa.matmul %3066, %3068 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %3070 = tosa.reshape %3069 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3071 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3072 = tosa.transpose %3070, %3071 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %3073 = tosa.reshape %3072 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %3074 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3075 = tosa.transpose %arg263, %3074 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3076 = tosa.reshape %3073 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_659 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3077 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3076, %3075 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_659 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3078 = tosa.reshape %3077 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3079 = tosa.add %2972, %3078 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3080 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_660 = arith.constant 2 : i32
    %3081 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3079 : tensor<1x40x1536xf32>) outs(%3080 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_660 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3082 = tosa.reduce_sum %3081 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3083 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3084 = tosa.reciprocal %3083 : (tensor<1xf32>) -> tensor<1xf32>
    %3085 = tosa.reshape %3084 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3086 = tosa.mul %3085, %3082 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3087 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3088 = tosa.add %3086, %3087 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3089 = tosa.rsqrt %3088 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3090 = tosa.mul %3079, %3089 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3091 = tosa.reshape %arg264 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3092 = tosa.mul %3091, %3090 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3093 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3094 = tosa.transpose %arg265, %3093 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3095 = tosa.reshape %3092 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_661 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3096 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3095, %3094 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_661 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3097 = tosa.reshape %3096 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3098 = tosa.sigmoid %3097 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3099 = tosa.mul %3097, %3098 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3100 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3101 = tosa.transpose %arg266, %3100 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3102 = tosa.reshape %3092 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_662 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3103 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3102, %3101 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_662 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3104 = tosa.reshape %3103 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3105 = tosa.mul %3099, %3104 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3106 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3107 = tosa.transpose %arg267, %3106 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %3108 = tosa.reshape %3105 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_663 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3109 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3108, %3107 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_663 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3110 = tosa.reshape %3109 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3111 = tosa.add %3079, %3110 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3112 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_664 = arith.constant 2 : i32
    %3113 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3111 : tensor<1x40x1536xf32>) outs(%3112 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_664 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3114 = tosa.reduce_sum %3113 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3115 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3116 = tosa.reciprocal %3115 : (tensor<1xf32>) -> tensor<1xf32>
    %3117 = tosa.reshape %3116 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3118 = tosa.mul %3117, %3114 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3119 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3120 = tosa.add %3118, %3119 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3121 = tosa.rsqrt %3120 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3122 = tosa.mul %3111, %3121 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3123 = tosa.reshape %arg268 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3124 = tosa.mul %3123, %3122 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3125 = tosa.reshape %3124 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3126 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3127 = tosa.transpose %arg269, %3126 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3128 = tosa.reshape %3125 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3129 = tosa.reshape %3127 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %3130 = tosa.matmul %3128, %3129 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %3131 = tosa.reshape %3130 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3132 = tosa.reshape %arg270 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %3133 = tosa.add %3132, %3131 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3134 = tosa.reshape %3133 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3135 = tosa.reshape %3124 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3136 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3137 = tosa.transpose %arg271, %3136 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3138 = tosa.reshape %3135 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3139 = tosa.reshape %3137 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3140 = tosa.matmul %3138, %3139 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3141 = tosa.reshape %3140 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3142 = tosa.reshape %arg272 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3143 = tosa.add %3142, %3141 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3144 = tosa.reshape %3143 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3145 = tosa.reshape %3124 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3146 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3147 = tosa.transpose %arg273, %3146 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3148 = tosa.reshape %3145 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3149 = tosa.reshape %3147 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3150 = tosa.matmul %3148, %3149 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3151 = tosa.reshape %3150 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3152 = tosa.reshape %arg274 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3153 = tosa.add %3152, %3151 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3154 = tosa.reshape %3153 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3155 = tosa.reshape %3134 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %3156 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3157 = tosa.transpose %3155, %3156 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %3158 = tosa.reshape %3144 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3159 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3160 = tosa.transpose %3158, %3159 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3161 = tosa.reshape %3154 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3162 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3163 = tosa.transpose %3161, %3162 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3164 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3165 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3166 = tosa.mul %3157, %3164 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_665 = tensor.extract_slice %3157[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_666 = tensor.extract_slice %3157[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %3167 = tensor.empty() : tensor<1x12x40x64xf32>
    %3168 = linalg.negf ins(%extracted_slice_666 : tensor<1x12x40x64xf32>) outs(%3167 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %3169 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_667 = tensor.insert_slice %3168 into %3169[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_668 = tensor.insert_slice %extracted_slice_665 into %inserted_slice_667[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %3170 = tosa.mul %inserted_slice_668, %3165 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3171 = tosa.add %3166, %3170 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3172 = tosa.mul %3160, %3164 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_669 = tensor.extract_slice %3160[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_670 = tensor.extract_slice %3160[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %3173 = tensor.empty() : tensor<1x2x40x64xf32>
    %3174 = linalg.negf ins(%extracted_slice_670 : tensor<1x2x40x64xf32>) outs(%3173 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %3175 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_671 = tensor.insert_slice %3174 into %3175[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_672 = tensor.insert_slice %extracted_slice_669 into %inserted_slice_671[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %3176 = tosa.mul %inserted_slice_672, %3165 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %3177 = tosa.add %3172, %3176 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_673 = tensor.extract_slice %3177[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_674 = tensor.extract_slice %extracted_slice_673[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3178 = tosa.reshape %extracted_slice_674 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_675 = tensor.extract_slice %3178[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_676 = tensor.extract_slice %extracted_slice_675[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3179 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3180 = tosa.add %extracted_slice_676, %3179 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3181 = tosa.identity %3180 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3182 = tosa.reshape %3181 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_677 = tensor.extract_slice %3163[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_678 = tensor.extract_slice %extracted_slice_677[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3183 = tosa.reshape %extracted_slice_678 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_679 = tensor.extract_slice %3183[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_680 = tensor.extract_slice %extracted_slice_679[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3184 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3185 = tosa.add %extracted_slice_680, %3184 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3186 = tosa.identity %3185 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3187 = tosa.reshape %3186 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_681 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_682 = tensor.extract_slice %extracted_slice_681[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_683 = tensor.extract_slice %extracted_slice_682[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_684 = arith.constant 0.000000e+00 : f32
    %splat_685 = tensor.splat %cst_684 : tensor<40x40xf32>
    %3188 = tosa.reshape %extracted_slice_683 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3189 = tosa.add %splat_685, %3188 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3190 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3191 = tosa.transpose %3182, %3190 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %3192 = tosa.reshape %3171 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3193 = tosa.reshape %3191 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %3194 = tosa.matmul %3192, %3193 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_686 = arith.constant 0.0883883461 : f32
    %splat_687 = tensor.splat %cst_686 : tensor<12x40x40xf32>
    %3195 = tosa.mul %3194, %splat_687 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %3196 = tosa.reshape %3189 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %3197 = tosa.add %3195, %3196 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %3198 = tosa.reduce_max %3197 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3199 = tosa.sub %3197, %3198 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3200 = math.exp %3199 : tensor<12x40x40xf32>
    %3201 = tosa.reduce_sum %3200 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3202 = tosa.log %3201 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3203 = tosa.add %3198, %3202 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3204 = tosa.sub %3197, %3203 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3205 = math.exp %3204 : tensor<12x40x40xf32>
    %3206 = tosa.reshape %3203 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %3207 = tosa.reshape %3187 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3208 = tosa.matmul %3205, %3207 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %3209 = tosa.reshape %3208 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3210 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3211 = tosa.transpose %3209, %3210 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %3212 = tosa.reshape %3211 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %3213 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3214 = tosa.transpose %arg275, %3213 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3215 = tosa.reshape %3212 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_688 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3216 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3215, %3214 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_688 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3217 = tosa.reshape %3216 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3218 = tosa.add %3111, %3217 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3219 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_689 = arith.constant 2 : i32
    %3220 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3218 : tensor<1x40x1536xf32>) outs(%3219 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_689 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3221 = tosa.reduce_sum %3220 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3222 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3223 = tosa.reciprocal %3222 : (tensor<1xf32>) -> tensor<1xf32>
    %3224 = tosa.reshape %3223 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3225 = tosa.mul %3224, %3221 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3226 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3227 = tosa.add %3225, %3226 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3228 = tosa.rsqrt %3227 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3229 = tosa.mul %3218, %3228 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3230 = tosa.reshape %arg276 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3231 = tosa.mul %3230, %3229 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3232 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3233 = tosa.transpose %arg277, %3232 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3234 = tosa.reshape %3231 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_690 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3235 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3234, %3233 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_690 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3236 = tosa.reshape %3235 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3237 = tosa.sigmoid %3236 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3238 = tosa.mul %3236, %3237 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3239 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3240 = tosa.transpose %arg278, %3239 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3241 = tosa.reshape %3231 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_691 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3242 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3241, %3240 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_691 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3243 = tosa.reshape %3242 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3244 = tosa.mul %3238, %3243 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3245 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3246 = tosa.transpose %arg279, %3245 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %3247 = tosa.reshape %3244 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_692 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3248 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3247, %3246 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_692 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3249 = tosa.reshape %3248 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3250 = tosa.add %3218, %3249 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3251 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_693 = arith.constant 2 : i32
    %3252 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3250 : tensor<1x40x1536xf32>) outs(%3251 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_693 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3253 = tosa.reduce_sum %3252 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3254 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3255 = tosa.reciprocal %3254 : (tensor<1xf32>) -> tensor<1xf32>
    %3256 = tosa.reshape %3255 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3257 = tosa.mul %3256, %3253 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3258 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3259 = tosa.add %3257, %3258 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3260 = tosa.rsqrt %3259 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3261 = tosa.mul %3250, %3260 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3262 = tosa.reshape %arg280 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3263 = tosa.mul %3262, %3261 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3264 = tosa.reshape %3263 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3265 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3266 = tosa.transpose %arg281, %3265 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3267 = tosa.reshape %3264 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3268 = tosa.reshape %3266 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %3269 = tosa.matmul %3267, %3268 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %3270 = tosa.reshape %3269 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3271 = tosa.reshape %arg282 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %3272 = tosa.add %3271, %3270 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3273 = tosa.reshape %3272 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3274 = tosa.reshape %3263 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3275 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3276 = tosa.transpose %arg283, %3275 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3277 = tosa.reshape %3274 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3278 = tosa.reshape %3276 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3279 = tosa.matmul %3277, %3278 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3280 = tosa.reshape %3279 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3281 = tosa.reshape %arg284 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3282 = tosa.add %3281, %3280 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3283 = tosa.reshape %3282 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3284 = tosa.reshape %3263 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3285 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3286 = tosa.transpose %arg285, %3285 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3287 = tosa.reshape %3284 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3288 = tosa.reshape %3286 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3289 = tosa.matmul %3287, %3288 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3290 = tosa.reshape %3289 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3291 = tosa.reshape %arg286 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3292 = tosa.add %3291, %3290 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3293 = tosa.reshape %3292 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3294 = tosa.reshape %3273 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %3295 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3296 = tosa.transpose %3294, %3295 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %3297 = tosa.reshape %3283 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3298 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3299 = tosa.transpose %3297, %3298 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3300 = tosa.reshape %3293 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3301 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3302 = tosa.transpose %3300, %3301 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3303 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3304 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3305 = tosa.mul %3296, %3303 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_694 = tensor.extract_slice %3296[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_695 = tensor.extract_slice %3296[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %3306 = tensor.empty() : tensor<1x12x40x64xf32>
    %3307 = linalg.negf ins(%extracted_slice_695 : tensor<1x12x40x64xf32>) outs(%3306 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %3308 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_696 = tensor.insert_slice %3307 into %3308[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_697 = tensor.insert_slice %extracted_slice_694 into %inserted_slice_696[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %3309 = tosa.mul %inserted_slice_697, %3304 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3310 = tosa.add %3305, %3309 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3311 = tosa.mul %3299, %3303 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_698 = tensor.extract_slice %3299[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_699 = tensor.extract_slice %3299[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %3312 = tensor.empty() : tensor<1x2x40x64xf32>
    %3313 = linalg.negf ins(%extracted_slice_699 : tensor<1x2x40x64xf32>) outs(%3312 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %3314 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_700 = tensor.insert_slice %3313 into %3314[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_701 = tensor.insert_slice %extracted_slice_698 into %inserted_slice_700[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %3315 = tosa.mul %inserted_slice_701, %3304 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %3316 = tosa.add %3311, %3315 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_702 = tensor.extract_slice %3316[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_703 = tensor.extract_slice %extracted_slice_702[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3317 = tosa.reshape %extracted_slice_703 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_704 = tensor.extract_slice %3317[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_705 = tensor.extract_slice %extracted_slice_704[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3318 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3319 = tosa.add %extracted_slice_705, %3318 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3320 = tosa.identity %3319 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3321 = tosa.reshape %3320 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_706 = tensor.extract_slice %3302[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_707 = tensor.extract_slice %extracted_slice_706[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3322 = tosa.reshape %extracted_slice_707 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_708 = tensor.extract_slice %3322[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_709 = tensor.extract_slice %extracted_slice_708[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3323 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3324 = tosa.add %extracted_slice_709, %3323 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3325 = tosa.identity %3324 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3326 = tosa.reshape %3325 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_710 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_711 = tensor.extract_slice %extracted_slice_710[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_712 = tensor.extract_slice %extracted_slice_711[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_713 = arith.constant 0.000000e+00 : f32
    %splat_714 = tensor.splat %cst_713 : tensor<40x40xf32>
    %3327 = tosa.reshape %extracted_slice_712 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3328 = tosa.add %splat_714, %3327 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3329 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3330 = tosa.transpose %3321, %3329 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %3331 = tosa.reshape %3310 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3332 = tosa.reshape %3330 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %3333 = tosa.matmul %3331, %3332 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_715 = arith.constant 0.0883883461 : f32
    %splat_716 = tensor.splat %cst_715 : tensor<12x40x40xf32>
    %3334 = tosa.mul %3333, %splat_716 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %3335 = tosa.reshape %3328 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %3336 = tosa.add %3334, %3335 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %3337 = tosa.reduce_max %3336 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3338 = tosa.sub %3336, %3337 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3339 = math.exp %3338 : tensor<12x40x40xf32>
    %3340 = tosa.reduce_sum %3339 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3341 = tosa.log %3340 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3342 = tosa.add %3337, %3341 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3343 = tosa.sub %3336, %3342 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3344 = math.exp %3343 : tensor<12x40x40xf32>
    %3345 = tosa.reshape %3342 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %3346 = tosa.reshape %3326 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3347 = tosa.matmul %3344, %3346 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %3348 = tosa.reshape %3347 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3349 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3350 = tosa.transpose %3348, %3349 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %3351 = tosa.reshape %3350 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %3352 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3353 = tosa.transpose %arg287, %3352 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3354 = tosa.reshape %3351 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_717 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3355 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3354, %3353 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_717 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3356 = tosa.reshape %3355 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3357 = tosa.add %3250, %3356 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3358 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_718 = arith.constant 2 : i32
    %3359 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3357 : tensor<1x40x1536xf32>) outs(%3358 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_718 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3360 = tosa.reduce_sum %3359 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3361 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3362 = tosa.reciprocal %3361 : (tensor<1xf32>) -> tensor<1xf32>
    %3363 = tosa.reshape %3362 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3364 = tosa.mul %3363, %3360 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3365 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3366 = tosa.add %3364, %3365 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3367 = tosa.rsqrt %3366 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3368 = tosa.mul %3357, %3367 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3369 = tosa.reshape %arg288 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3370 = tosa.mul %3369, %3368 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3371 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3372 = tosa.transpose %arg289, %3371 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3373 = tosa.reshape %3370 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_719 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3374 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3373, %3372 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_719 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3375 = tosa.reshape %3374 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3376 = tosa.sigmoid %3375 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3377 = tosa.mul %3375, %3376 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3378 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3379 = tosa.transpose %arg290, %3378 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3380 = tosa.reshape %3370 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_720 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3381 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3380, %3379 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_720 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3382 = tosa.reshape %3381 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3383 = tosa.mul %3377, %3382 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3384 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3385 = tosa.transpose %arg291, %3384 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %3386 = tosa.reshape %3383 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_721 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3387 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3386, %3385 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_721 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3388 = tosa.reshape %3387 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3389 = tosa.add %3357, %3388 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3390 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_722 = arith.constant 2 : i32
    %3391 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3389 : tensor<1x40x1536xf32>) outs(%3390 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_722 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3392 = tosa.reduce_sum %3391 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3393 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3394 = tosa.reciprocal %3393 : (tensor<1xf32>) -> tensor<1xf32>
    %3395 = tosa.reshape %3394 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3396 = tosa.mul %3395, %3392 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3397 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3398 = tosa.add %3396, %3397 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3399 = tosa.rsqrt %3398 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3400 = tosa.mul %3389, %3399 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3401 = tosa.reshape %arg292 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3402 = tosa.mul %3401, %3400 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3403 = tosa.reshape %3402 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3404 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3405 = tosa.transpose %arg293, %3404 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3406 = tosa.reshape %3403 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3407 = tosa.reshape %3405 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %3408 = tosa.matmul %3406, %3407 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %3409 = tosa.reshape %3408 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3410 = tosa.reshape %arg294 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %3411 = tosa.add %3410, %3409 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3412 = tosa.reshape %3411 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3413 = tosa.reshape %3402 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3414 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3415 = tosa.transpose %arg295, %3414 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3416 = tosa.reshape %3413 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3417 = tosa.reshape %3415 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3418 = tosa.matmul %3416, %3417 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3419 = tosa.reshape %3418 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3420 = tosa.reshape %arg296 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3421 = tosa.add %3420, %3419 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3422 = tosa.reshape %3421 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3423 = tosa.reshape %3402 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3424 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3425 = tosa.transpose %arg297, %3424 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3426 = tosa.reshape %3423 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3427 = tosa.reshape %3425 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3428 = tosa.matmul %3426, %3427 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3429 = tosa.reshape %3428 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3430 = tosa.reshape %arg298 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3431 = tosa.add %3430, %3429 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3432 = tosa.reshape %3431 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3433 = tosa.reshape %3412 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %3434 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3435 = tosa.transpose %3433, %3434 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %3436 = tosa.reshape %3422 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3437 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3438 = tosa.transpose %3436, %3437 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3439 = tosa.reshape %3432 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3440 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3441 = tosa.transpose %3439, %3440 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3442 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3443 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3444 = tosa.mul %3435, %3442 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_723 = tensor.extract_slice %3435[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_724 = tensor.extract_slice %3435[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %3445 = tensor.empty() : tensor<1x12x40x64xf32>
    %3446 = linalg.negf ins(%extracted_slice_724 : tensor<1x12x40x64xf32>) outs(%3445 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %3447 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_725 = tensor.insert_slice %3446 into %3447[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_726 = tensor.insert_slice %extracted_slice_723 into %inserted_slice_725[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %3448 = tosa.mul %inserted_slice_726, %3443 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3449 = tosa.add %3444, %3448 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3450 = tosa.mul %3438, %3442 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_727 = tensor.extract_slice %3438[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_728 = tensor.extract_slice %3438[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %3451 = tensor.empty() : tensor<1x2x40x64xf32>
    %3452 = linalg.negf ins(%extracted_slice_728 : tensor<1x2x40x64xf32>) outs(%3451 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %3453 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_729 = tensor.insert_slice %3452 into %3453[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_730 = tensor.insert_slice %extracted_slice_727 into %inserted_slice_729[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %3454 = tosa.mul %inserted_slice_730, %3443 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %3455 = tosa.add %3450, %3454 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_731 = tensor.extract_slice %3455[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_732 = tensor.extract_slice %extracted_slice_731[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3456 = tosa.reshape %extracted_slice_732 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_733 = tensor.extract_slice %3456[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_734 = tensor.extract_slice %extracted_slice_733[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3457 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3458 = tosa.add %extracted_slice_734, %3457 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3459 = tosa.identity %3458 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3460 = tosa.reshape %3459 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_735 = tensor.extract_slice %3441[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_736 = tensor.extract_slice %extracted_slice_735[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3461 = tosa.reshape %extracted_slice_736 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_737 = tensor.extract_slice %3461[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_738 = tensor.extract_slice %extracted_slice_737[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3462 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3463 = tosa.add %extracted_slice_738, %3462 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3464 = tosa.identity %3463 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3465 = tosa.reshape %3464 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_739 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_740 = tensor.extract_slice %extracted_slice_739[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_741 = tensor.extract_slice %extracted_slice_740[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_742 = arith.constant 0.000000e+00 : f32
    %splat_743 = tensor.splat %cst_742 : tensor<40x40xf32>
    %3466 = tosa.reshape %extracted_slice_741 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3467 = tosa.add %splat_743, %3466 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3468 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3469 = tosa.transpose %3460, %3468 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %3470 = tosa.reshape %3449 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3471 = tosa.reshape %3469 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %3472 = tosa.matmul %3470, %3471 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_744 = arith.constant 0.0883883461 : f32
    %splat_745 = tensor.splat %cst_744 : tensor<12x40x40xf32>
    %3473 = tosa.mul %3472, %splat_745 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %3474 = tosa.reshape %3467 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %3475 = tosa.add %3473, %3474 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %3476 = tosa.reduce_max %3475 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3477 = tosa.sub %3475, %3476 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3478 = math.exp %3477 : tensor<12x40x40xf32>
    %3479 = tosa.reduce_sum %3478 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3480 = tosa.log %3479 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3481 = tosa.add %3476, %3480 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3482 = tosa.sub %3475, %3481 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3483 = math.exp %3482 : tensor<12x40x40xf32>
    %3484 = tosa.reshape %3481 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %3485 = tosa.reshape %3465 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3486 = tosa.matmul %3483, %3485 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %3487 = tosa.reshape %3486 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3488 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3489 = tosa.transpose %3487, %3488 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %3490 = tosa.reshape %3489 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %3491 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3492 = tosa.transpose %arg299, %3491 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3493 = tosa.reshape %3490 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_746 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3494 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3493, %3492 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_746 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3495 = tosa.reshape %3494 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3496 = tosa.add %3389, %3495 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3497 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_747 = arith.constant 2 : i32
    %3498 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3496 : tensor<1x40x1536xf32>) outs(%3497 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_747 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3499 = tosa.reduce_sum %3498 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3500 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3501 = tosa.reciprocal %3500 : (tensor<1xf32>) -> tensor<1xf32>
    %3502 = tosa.reshape %3501 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3503 = tosa.mul %3502, %3499 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3504 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3505 = tosa.add %3503, %3504 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3506 = tosa.rsqrt %3505 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3507 = tosa.mul %3496, %3506 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3508 = tosa.reshape %arg300 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3509 = tosa.mul %3508, %3507 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3510 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3511 = tosa.transpose %arg301, %3510 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3512 = tosa.reshape %3509 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_748 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3513 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3512, %3511 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_748 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3514 = tosa.reshape %3513 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3515 = tosa.sigmoid %3514 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3516 = tosa.mul %3514, %3515 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3517 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3518 = tosa.transpose %arg302, %3517 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3519 = tosa.reshape %3509 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_749 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3520 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3519, %3518 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_749 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3521 = tosa.reshape %3520 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3522 = tosa.mul %3516, %3521 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3523 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3524 = tosa.transpose %arg303, %3523 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %3525 = tosa.reshape %3522 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_750 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3526 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3525, %3524 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_750 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3527 = tosa.reshape %3526 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3528 = tosa.add %3496, %3527 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3529 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_751 = arith.constant 2 : i32
    %3530 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3528 : tensor<1x40x1536xf32>) outs(%3529 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_751 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3531 = tosa.reduce_sum %3530 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3532 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3533 = tosa.reciprocal %3532 : (tensor<1xf32>) -> tensor<1xf32>
    %3534 = tosa.reshape %3533 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3535 = tosa.mul %3534, %3531 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3536 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3537 = tosa.add %3535, %3536 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3538 = tosa.rsqrt %3537 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3539 = tosa.mul %3528, %3538 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3540 = tosa.reshape %arg304 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3541 = tosa.mul %3540, %3539 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3542 = tosa.reshape %3541 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3543 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3544 = tosa.transpose %arg305, %3543 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3545 = tosa.reshape %3542 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3546 = tosa.reshape %3544 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %3547 = tosa.matmul %3545, %3546 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %3548 = tosa.reshape %3547 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3549 = tosa.reshape %arg306 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %3550 = tosa.add %3549, %3548 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3551 = tosa.reshape %3550 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3552 = tosa.reshape %3541 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3553 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3554 = tosa.transpose %arg307, %3553 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3555 = tosa.reshape %3552 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3556 = tosa.reshape %3554 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3557 = tosa.matmul %3555, %3556 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3558 = tosa.reshape %3557 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3559 = tosa.reshape %arg308 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3560 = tosa.add %3559, %3558 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3561 = tosa.reshape %3560 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3562 = tosa.reshape %3541 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3563 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3564 = tosa.transpose %arg309, %3563 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3565 = tosa.reshape %3562 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3566 = tosa.reshape %3564 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3567 = tosa.matmul %3565, %3566 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3568 = tosa.reshape %3567 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3569 = tosa.reshape %arg310 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3570 = tosa.add %3569, %3568 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3571 = tosa.reshape %3570 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3572 = tosa.reshape %3551 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %3573 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3574 = tosa.transpose %3572, %3573 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %3575 = tosa.reshape %3561 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3576 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3577 = tosa.transpose %3575, %3576 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3578 = tosa.reshape %3571 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3579 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3580 = tosa.transpose %3578, %3579 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3581 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3582 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3583 = tosa.mul %3574, %3581 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_752 = tensor.extract_slice %3574[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_753 = tensor.extract_slice %3574[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %3584 = tensor.empty() : tensor<1x12x40x64xf32>
    %3585 = linalg.negf ins(%extracted_slice_753 : tensor<1x12x40x64xf32>) outs(%3584 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %3586 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_754 = tensor.insert_slice %3585 into %3586[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_755 = tensor.insert_slice %extracted_slice_752 into %inserted_slice_754[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %3587 = tosa.mul %inserted_slice_755, %3582 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3588 = tosa.add %3583, %3587 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3589 = tosa.mul %3577, %3581 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_756 = tensor.extract_slice %3577[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_757 = tensor.extract_slice %3577[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %3590 = tensor.empty() : tensor<1x2x40x64xf32>
    %3591 = linalg.negf ins(%extracted_slice_757 : tensor<1x2x40x64xf32>) outs(%3590 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %3592 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_758 = tensor.insert_slice %3591 into %3592[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_759 = tensor.insert_slice %extracted_slice_756 into %inserted_slice_758[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %3593 = tosa.mul %inserted_slice_759, %3582 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %3594 = tosa.add %3589, %3593 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_760 = tensor.extract_slice %3594[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_761 = tensor.extract_slice %extracted_slice_760[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3595 = tosa.reshape %extracted_slice_761 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_762 = tensor.extract_slice %3595[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_763 = tensor.extract_slice %extracted_slice_762[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3596 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3597 = tosa.add %extracted_slice_763, %3596 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3598 = tosa.identity %3597 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3599 = tosa.reshape %3598 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_764 = tensor.extract_slice %3580[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_765 = tensor.extract_slice %extracted_slice_764[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3600 = tosa.reshape %extracted_slice_765 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_766 = tensor.extract_slice %3600[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_767 = tensor.extract_slice %extracted_slice_766[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3601 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3602 = tosa.add %extracted_slice_767, %3601 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3603 = tosa.identity %3602 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3604 = tosa.reshape %3603 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_768 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_769 = tensor.extract_slice %extracted_slice_768[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_770 = tensor.extract_slice %extracted_slice_769[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_771 = arith.constant 0.000000e+00 : f32
    %splat_772 = tensor.splat %cst_771 : tensor<40x40xf32>
    %3605 = tosa.reshape %extracted_slice_770 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3606 = tosa.add %splat_772, %3605 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3607 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3608 = tosa.transpose %3599, %3607 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %3609 = tosa.reshape %3588 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3610 = tosa.reshape %3608 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %3611 = tosa.matmul %3609, %3610 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_773 = arith.constant 0.0883883461 : f32
    %splat_774 = tensor.splat %cst_773 : tensor<12x40x40xf32>
    %3612 = tosa.mul %3611, %splat_774 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %3613 = tosa.reshape %3606 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %3614 = tosa.add %3612, %3613 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %3615 = tosa.reduce_max %3614 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3616 = tosa.sub %3614, %3615 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3617 = math.exp %3616 : tensor<12x40x40xf32>
    %3618 = tosa.reduce_sum %3617 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3619 = tosa.log %3618 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3620 = tosa.add %3615, %3619 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3621 = tosa.sub %3614, %3620 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3622 = math.exp %3621 : tensor<12x40x40xf32>
    %3623 = tosa.reshape %3620 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %3624 = tosa.reshape %3604 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3625 = tosa.matmul %3622, %3624 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %3626 = tosa.reshape %3625 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3627 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3628 = tosa.transpose %3626, %3627 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %3629 = tosa.reshape %3628 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %3630 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3631 = tosa.transpose %arg311, %3630 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3632 = tosa.reshape %3629 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_775 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3633 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3632, %3631 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_775 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3634 = tosa.reshape %3633 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3635 = tosa.add %3528, %3634 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3636 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_776 = arith.constant 2 : i32
    %3637 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3635 : tensor<1x40x1536xf32>) outs(%3636 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_776 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3638 = tosa.reduce_sum %3637 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3639 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3640 = tosa.reciprocal %3639 : (tensor<1xf32>) -> tensor<1xf32>
    %3641 = tosa.reshape %3640 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3642 = tosa.mul %3641, %3638 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3643 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3644 = tosa.add %3642, %3643 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3645 = tosa.rsqrt %3644 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3646 = tosa.mul %3635, %3645 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3647 = tosa.reshape %arg312 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3648 = tosa.mul %3647, %3646 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3649 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3650 = tosa.transpose %arg313, %3649 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3651 = tosa.reshape %3648 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_777 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3652 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3651, %3650 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_777 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3653 = tosa.reshape %3652 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3654 = tosa.sigmoid %3653 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3655 = tosa.mul %3653, %3654 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3656 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3657 = tosa.transpose %arg314, %3656 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3658 = tosa.reshape %3648 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_778 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3659 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3658, %3657 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_778 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3660 = tosa.reshape %3659 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3661 = tosa.mul %3655, %3660 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3662 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3663 = tosa.transpose %arg315, %3662 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %3664 = tosa.reshape %3661 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_779 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3665 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3664, %3663 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_779 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3666 = tosa.reshape %3665 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3667 = tosa.add %3635, %3666 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3668 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_780 = arith.constant 2 : i32
    %3669 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3667 : tensor<1x40x1536xf32>) outs(%3668 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_780 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3670 = tosa.reduce_sum %3669 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3671 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3672 = tosa.reciprocal %3671 : (tensor<1xf32>) -> tensor<1xf32>
    %3673 = tosa.reshape %3672 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3674 = tosa.mul %3673, %3670 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3675 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3676 = tosa.add %3674, %3675 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3677 = tosa.rsqrt %3676 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3678 = tosa.mul %3667, %3677 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3679 = tosa.reshape %arg316 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3680 = tosa.mul %3679, %3678 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3681 = tosa.reshape %3680 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3682 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3683 = tosa.transpose %arg317, %3682 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3684 = tosa.reshape %3681 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3685 = tosa.reshape %3683 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %3686 = tosa.matmul %3684, %3685 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %3687 = tosa.reshape %3686 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3688 = tosa.reshape %arg318 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %3689 = tosa.add %3688, %3687 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3690 = tosa.reshape %3689 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3691 = tosa.reshape %3680 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3692 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3693 = tosa.transpose %arg319, %3692 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3694 = tosa.reshape %3691 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3695 = tosa.reshape %3693 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3696 = tosa.matmul %3694, %3695 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3697 = tosa.reshape %3696 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3698 = tosa.reshape %arg320 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3699 = tosa.add %3698, %3697 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3700 = tosa.reshape %3699 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3701 = tosa.reshape %3680 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3702 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3703 = tosa.transpose %arg321, %3702 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3704 = tosa.reshape %3701 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3705 = tosa.reshape %3703 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3706 = tosa.matmul %3704, %3705 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3707 = tosa.reshape %3706 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3708 = tosa.reshape %arg322 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3709 = tosa.add %3708, %3707 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3710 = tosa.reshape %3709 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3711 = tosa.reshape %3690 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %3712 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3713 = tosa.transpose %3711, %3712 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %3714 = tosa.reshape %3700 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3715 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3716 = tosa.transpose %3714, %3715 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3717 = tosa.reshape %3710 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3718 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3719 = tosa.transpose %3717, %3718 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3720 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3721 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3722 = tosa.mul %3713, %3720 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_781 = tensor.extract_slice %3713[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_782 = tensor.extract_slice %3713[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %3723 = tensor.empty() : tensor<1x12x40x64xf32>
    %3724 = linalg.negf ins(%extracted_slice_782 : tensor<1x12x40x64xf32>) outs(%3723 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %3725 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_783 = tensor.insert_slice %3724 into %3725[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_784 = tensor.insert_slice %extracted_slice_781 into %inserted_slice_783[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %3726 = tosa.mul %inserted_slice_784, %3721 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3727 = tosa.add %3722, %3726 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3728 = tosa.mul %3716, %3720 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_785 = tensor.extract_slice %3716[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_786 = tensor.extract_slice %3716[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %3729 = tensor.empty() : tensor<1x2x40x64xf32>
    %3730 = linalg.negf ins(%extracted_slice_786 : tensor<1x2x40x64xf32>) outs(%3729 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %3731 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_787 = tensor.insert_slice %3730 into %3731[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_788 = tensor.insert_slice %extracted_slice_785 into %inserted_slice_787[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %3732 = tosa.mul %inserted_slice_788, %3721 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %3733 = tosa.add %3728, %3732 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_789 = tensor.extract_slice %3733[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_790 = tensor.extract_slice %extracted_slice_789[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3734 = tosa.reshape %extracted_slice_790 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_791 = tensor.extract_slice %3734[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_792 = tensor.extract_slice %extracted_slice_791[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3735 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3736 = tosa.add %extracted_slice_792, %3735 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3737 = tosa.identity %3736 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3738 = tosa.reshape %3737 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_793 = tensor.extract_slice %3719[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_794 = tensor.extract_slice %extracted_slice_793[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3739 = tosa.reshape %extracted_slice_794 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_795 = tensor.extract_slice %3739[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_796 = tensor.extract_slice %extracted_slice_795[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3740 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3741 = tosa.add %extracted_slice_796, %3740 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3742 = tosa.identity %3741 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3743 = tosa.reshape %3742 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_797 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_798 = tensor.extract_slice %extracted_slice_797[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_799 = tensor.extract_slice %extracted_slice_798[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_800 = arith.constant 0.000000e+00 : f32
    %splat_801 = tensor.splat %cst_800 : tensor<40x40xf32>
    %3744 = tosa.reshape %extracted_slice_799 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3745 = tosa.add %splat_801, %3744 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3746 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3747 = tosa.transpose %3738, %3746 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %3748 = tosa.reshape %3727 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3749 = tosa.reshape %3747 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %3750 = tosa.matmul %3748, %3749 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_802 = arith.constant 0.0883883461 : f32
    %splat_803 = tensor.splat %cst_802 : tensor<12x40x40xf32>
    %3751 = tosa.mul %3750, %splat_803 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %3752 = tosa.reshape %3745 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %3753 = tosa.add %3751, %3752 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %3754 = tosa.reduce_max %3753 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3755 = tosa.sub %3753, %3754 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3756 = math.exp %3755 : tensor<12x40x40xf32>
    %3757 = tosa.reduce_sum %3756 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3758 = tosa.log %3757 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3759 = tosa.add %3754, %3758 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3760 = tosa.sub %3753, %3759 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3761 = math.exp %3760 : tensor<12x40x40xf32>
    %3762 = tosa.reshape %3759 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %3763 = tosa.reshape %3743 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3764 = tosa.matmul %3761, %3763 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %3765 = tosa.reshape %3764 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3766 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3767 = tosa.transpose %3765, %3766 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %3768 = tosa.reshape %3767 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %3769 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3770 = tosa.transpose %arg323, %3769 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3771 = tosa.reshape %3768 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_804 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3772 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3771, %3770 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_804 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3773 = tosa.reshape %3772 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3774 = tosa.add %3667, %3773 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3775 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_805 = arith.constant 2 : i32
    %3776 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3774 : tensor<1x40x1536xf32>) outs(%3775 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_805 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3777 = tosa.reduce_sum %3776 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3778 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3779 = tosa.reciprocal %3778 : (tensor<1xf32>) -> tensor<1xf32>
    %3780 = tosa.reshape %3779 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3781 = tosa.mul %3780, %3777 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3782 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3783 = tosa.add %3781, %3782 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3784 = tosa.rsqrt %3783 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3785 = tosa.mul %3774, %3784 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3786 = tosa.reshape %arg324 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3787 = tosa.mul %3786, %3785 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3788 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3789 = tosa.transpose %arg325, %3788 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3790 = tosa.reshape %3787 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_806 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3791 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3790, %3789 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_806 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3792 = tosa.reshape %3791 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3793 = tosa.sigmoid %3792 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3794 = tosa.mul %3792, %3793 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3795 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3796 = tosa.transpose %arg326, %3795 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3797 = tosa.reshape %3787 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_807 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3798 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3797, %3796 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_807 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3799 = tosa.reshape %3798 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3800 = tosa.mul %3794, %3799 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3801 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3802 = tosa.transpose %arg327, %3801 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %3803 = tosa.reshape %3800 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_808 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3804 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3803, %3802 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_808 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3805 = tosa.reshape %3804 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3806 = tosa.add %3774, %3805 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3807 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_809 = arith.constant 2 : i32
    %3808 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3806 : tensor<1x40x1536xf32>) outs(%3807 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_809 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3809 = tosa.reduce_sum %3808 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3810 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3811 = tosa.reciprocal %3810 : (tensor<1xf32>) -> tensor<1xf32>
    %3812 = tosa.reshape %3811 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3813 = tosa.mul %3812, %3809 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3814 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3815 = tosa.add %3813, %3814 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3816 = tosa.rsqrt %3815 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3817 = tosa.mul %3806, %3816 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3818 = tosa.reshape %arg328 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3819 = tosa.mul %3818, %3817 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3820 = tosa.reshape %3819 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3821 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3822 = tosa.transpose %arg329, %3821 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3823 = tosa.reshape %3820 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3824 = tosa.reshape %3822 {new_shape = array<i64: 1, 1536, 1536>} : (tensor<1536x1536xf32>) -> tensor<1x1536x1536xf32>
    %3825 = tosa.matmul %3823, %3824 : (tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) -> tensor<1x40x1536xf32>
    %3826 = tosa.reshape %3825 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3827 = tosa.reshape %arg330 {new_shape = array<i64: 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %3828 = tosa.add %3827, %3826 : (tensor<1x1536xf32>, tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3829 = tosa.reshape %3828 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3830 = tosa.reshape %3819 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3831 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3832 = tosa.transpose %arg331, %3831 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3833 = tosa.reshape %3830 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3834 = tosa.reshape %3832 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3835 = tosa.matmul %3833, %3834 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3836 = tosa.reshape %3835 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3837 = tosa.reshape %arg332 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3838 = tosa.add %3837, %3836 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3839 = tosa.reshape %3838 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3840 = tosa.reshape %3819 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %3841 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3842 = tosa.transpose %arg333, %3841 : (tensor<256x1536xf32>, tensor<2xi32>) -> tensor<1536x256xf32>
    %3843 = tosa.reshape %3840 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3844 = tosa.reshape %3842 {new_shape = array<i64: 1, 1536, 256>} : (tensor<1536x256xf32>) -> tensor<1x1536x256xf32>
    %3845 = tosa.matmul %3843, %3844 : (tensor<1x40x1536xf32>, tensor<1x1536x256xf32>) -> tensor<1x40x256xf32>
    %3846 = tosa.reshape %3845 {new_shape = array<i64: 40, 256>} : (tensor<1x40x256xf32>) -> tensor<40x256xf32>
    %3847 = tosa.reshape %arg334 {new_shape = array<i64: 1, 256>} : (tensor<256xf32>) -> tensor<1x256xf32>
    %3848 = tosa.add %3847, %3846 : (tensor<1x256xf32>, tensor<40x256xf32>) -> tensor<40x256xf32>
    %3849 = tosa.reshape %3848 {new_shape = array<i64: 1, 40, 256>} : (tensor<40x256xf32>) -> tensor<1x40x256xf32>
    %3850 = tosa.reshape %3829 {new_shape = array<i64: 1, 40, 12, 128>} : (tensor<1x40x1536xf32>) -> tensor<1x40x12x128xf32>
    %3851 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3852 = tosa.transpose %3850, %3851 : (tensor<1x40x12x128xf32>, tensor<4xi32>) -> tensor<1x12x40x128xf32>
    %3853 = tosa.reshape %3839 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3854 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3855 = tosa.transpose %3853, %3854 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3856 = tosa.reshape %3849 {new_shape = array<i64: 1, 40, 2, 128>} : (tensor<1x40x256xf32>) -> tensor<1x40x2x128xf32>
    %3857 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3858 = tosa.transpose %3856, %3857 : (tensor<1x40x2x128xf32>, tensor<4xi32>) -> tensor<1x2x40x128xf32>
    %3859 = tosa.reshape %51 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3860 = tosa.reshape %53 {new_shape = array<i64: 1, 1, 40, 128>} : (tensor<1x40x128xf32>) -> tensor<1x1x40x128xf32>
    %3861 = tosa.mul %3852, %3859 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_810 = tensor.extract_slice %3852[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %extracted_slice_811 = tensor.extract_slice %3852[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x128xf32> to tensor<1x12x40x64xf32>
    %3862 = tensor.empty() : tensor<1x12x40x64xf32>
    %3863 = linalg.negf ins(%extracted_slice_811 : tensor<1x12x40x64xf32>) outs(%3862 : tensor<1x12x40x64xf32>) -> tensor<1x12x40x64xf32>
    %3864 = tensor.empty() : tensor<1x12x40x128xf32>
    %inserted_slice_812 = tensor.insert_slice %3863 into %3864[0, 0, 0, 0] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %inserted_slice_813 = tensor.insert_slice %extracted_slice_810 into %inserted_slice_812[0, 0, 0, 64] [1, 12, 40, 64] [1, 1, 1, 1] : tensor<1x12x40x64xf32> into tensor<1x12x40x128xf32>
    %3865 = tosa.mul %inserted_slice_813, %3860 : (tensor<1x12x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3866 = tosa.add %3861, %3865 : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3867 = tosa.mul %3855, %3859 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_814 = tensor.extract_slice %3855[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %extracted_slice_815 = tensor.extract_slice %3855[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x64xf32>
    %3868 = tensor.empty() : tensor<1x2x40x64xf32>
    %3869 = linalg.negf ins(%extracted_slice_815 : tensor<1x2x40x64xf32>) outs(%3868 : tensor<1x2x40x64xf32>) -> tensor<1x2x40x64xf32>
    %3870 = tensor.empty() : tensor<1x2x40x128xf32>
    %inserted_slice_816 = tensor.insert_slice %3869 into %3870[0, 0, 0, 0] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %inserted_slice_817 = tensor.insert_slice %extracted_slice_814 into %inserted_slice_816[0, 0, 0, 64] [1, 2, 40, 64] [1, 1, 1, 1] : tensor<1x2x40x64xf32> into tensor<1x2x40x128xf32>
    %3871 = tosa.mul %inserted_slice_817, %3860 : (tensor<1x2x40x128xf32>, tensor<1x1x40x128xf32>) -> tensor<1x2x40x128xf32>
    %3872 = tosa.add %3867, %3871 : (tensor<1x2x40x128xf32>, tensor<1x2x40x128xf32>) -> tensor<1x2x40x128xf32>
    %extracted_slice_818 = tensor.extract_slice %3872[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_819 = tensor.extract_slice %extracted_slice_818[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3873 = tosa.reshape %extracted_slice_819 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_820 = tensor.extract_slice %3873[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_821 = tensor.extract_slice %extracted_slice_820[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3874 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3875 = tosa.add %extracted_slice_821, %3874 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3876 = tosa.identity %3875 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3877 = tosa.reshape %3876 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_822 = tensor.extract_slice %3858[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %extracted_slice_823 = tensor.extract_slice %extracted_slice_822[0, 0, 0, 0] [1, 2, 40, 128] [1, 1, 1, 1] : tensor<1x2x40x128xf32> to tensor<1x2x40x128xf32>
    %3878 = tosa.reshape %extracted_slice_823 {new_shape = array<i64: 1, 2, 1, 40, 128>} : (tensor<1x2x40x128xf32>) -> tensor<1x2x1x40x128xf32>
    %extracted_slice_824 = tensor.extract_slice %3878[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %extracted_slice_825 = tensor.extract_slice %extracted_slice_824[0, 0, 0, 0, 0] [1, 2, 1, 40, 128] [1, 1, 1, 1, 1] : tensor<1x2x1x40x128xf32> to tensor<1x2x1x40x128xf32>
    %3879 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %3880 = tosa.add %extracted_slice_825, %3879 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3881 = tosa.identity %3880 : (tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %3882 = tosa.reshape %3881 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<1x2x6x40x128xf32>) -> tensor<1x12x40x128xf32>
    %extracted_slice_826 = tensor.extract_slice %inserted_slice_21[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_827 = tensor.extract_slice %extracted_slice_826[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %extracted_slice_828 = tensor.extract_slice %extracted_slice_827[0, 0, 0, 0] [1, 1, 40, 40] [1, 1, 1, 1] : tensor<1x1x40x40xf32> to tensor<1x1x40x40xf32>
    %cst_829 = arith.constant 0.000000e+00 : f32
    %splat_830 = tensor.splat %cst_829 : tensor<40x40xf32>
    %3883 = tosa.reshape %extracted_slice_828 {new_shape = array<i64: 40, 40>} : (tensor<1x1x40x40xf32>) -> tensor<40x40xf32>
    %3884 = tosa.add %splat_830, %3883 : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>
    %3885 = "tosa.const"() <{value = dense<[0, 1, 3, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3886 = tosa.transpose %3877, %3885 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x12x128x40xf32>
    %3887 = tosa.reshape %3866 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3888 = tosa.reshape %3886 {new_shape = array<i64: 12, 128, 40>} : (tensor<1x12x128x40xf32>) -> tensor<12x128x40xf32>
    %3889 = tosa.matmul %3887, %3888 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>) -> tensor<12x40x40xf32>
    %cst_831 = arith.constant 0.0883883461 : f32
    %splat_832 = tensor.splat %cst_831 : tensor<12x40x40xf32>
    %3890 = tosa.mul %3889, %splat_832 : (tensor<12x40x40xf32>, tensor<12x40x40xf32>) -> tensor<12x40x40xf32>
    %3891 = tosa.reshape %3884 {new_shape = array<i64: 1, 40, 40>} : (tensor<40x40xf32>) -> tensor<1x40x40xf32>
    %3892 = tosa.add %3890, %3891 : (tensor<12x40x40xf32>, tensor<1x40x40xf32>) -> tensor<12x40x40xf32>
    %3893 = tosa.reduce_max %3892 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3894 = tosa.sub %3892, %3893 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3895 = math.exp %3894 : tensor<12x40x40xf32>
    %3896 = tosa.reduce_sum %3895 {axis = 2 : i32} : (tensor<12x40x40xf32>) -> tensor<12x40x1xf32>
    %3897 = tosa.log %3896 : (tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3898 = tosa.add %3893, %3897 : (tensor<12x40x1xf32>, tensor<12x40x1xf32>) -> tensor<12x40x1xf32>
    %3899 = tosa.sub %3892, %3898 : (tensor<12x40x40xf32>, tensor<12x40x1xf32>) -> tensor<12x40x40xf32>
    %3900 = math.exp %3899 : tensor<12x40x40xf32>
    %3901 = tosa.reshape %3898 {new_shape = array<i64: 1, 12, 40>} : (tensor<12x40x1xf32>) -> tensor<1x12x40xf32>
    %3902 = tosa.reshape %3882 {new_shape = array<i64: 12, 40, 128>} : (tensor<1x12x40x128xf32>) -> tensor<12x40x128xf32>
    %3903 = tosa.matmul %3900, %3902 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>) -> tensor<12x40x128xf32>
    %3904 = tosa.reshape %3903 {new_shape = array<i64: 1, 12, 40, 128>} : (tensor<12x40x128xf32>) -> tensor<1x12x40x128xf32>
    %3905 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %3906 = tosa.transpose %3904, %3905 : (tensor<1x12x40x128xf32>, tensor<4xi32>) -> tensor<1x40x12x128xf32>
    %3907 = tosa.reshape %3906 {new_shape = array<i64: 1, 40, 1536>} : (tensor<1x40x12x128xf32>) -> tensor<1x40x1536xf32>
    %3908 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3909 = tosa.transpose %arg335, %3908 : (tensor<1536x1536xf32>, tensor<2xi32>) -> tensor<1536x1536xf32>
    %3910 = tosa.reshape %3907 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_833 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3911 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3910, %3909 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_833 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3912 = tosa.reshape %3911 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3913 = tosa.add %3806, %3912 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3914 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_834 = arith.constant 2 : i32
    %3915 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3913 : tensor<1x40x1536xf32>) outs(%3914 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_834 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3916 = tosa.reduce_sum %3915 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3917 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3918 = tosa.reciprocal %3917 : (tensor<1xf32>) -> tensor<1xf32>
    %3919 = tosa.reshape %3918 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3920 = tosa.mul %3919, %3916 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3921 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3922 = tosa.add %3920, %3921 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3923 = tosa.rsqrt %3922 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3924 = tosa.mul %3913, %3923 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3925 = tosa.reshape %arg336 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3926 = tosa.mul %3925, %3924 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3927 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3928 = tosa.transpose %arg337, %3927 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3929 = tosa.reshape %3926 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_835 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3930 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3929, %3928 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_835 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3931 = tosa.reshape %3930 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3932 = tosa.sigmoid %3931 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3933 = tosa.mul %3931, %3932 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3934 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3935 = tosa.transpose %arg338, %3934 : (tensor<8960x1536xf32>, tensor<2xi32>) -> tensor<1536x8960xf32>
    %3936 = tosa.reshape %3926 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_836 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %3937 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3936, %3935 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_836 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %3938 = tosa.reshape %3937 {new_shape = array<i64: 1, 40, 8960>} : (tensor<40x8960xf32>) -> tensor<1x40x8960xf32>
    %3939 = tosa.mul %3933, %3938 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %3940 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3941 = tosa.transpose %arg339, %3940 : (tensor<1536x8960xf32>, tensor<2xi32>) -> tensor<8960x1536xf32>
    %3942 = tosa.reshape %3939 {new_shape = array<i64: 40, 8960>} : (tensor<1x40x8960xf32>) -> tensor<40x8960xf32>
    %cst_837 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %3943 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3942, %3941 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_837 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %3944 = tosa.reshape %3943 {new_shape = array<i64: 1, 40, 1536>} : (tensor<40x1536xf32>) -> tensor<1x40x1536xf32>
    %3945 = tosa.add %3913, %3944 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %3946 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_838 = arith.constant 2 : i32
    %3947 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3945 : tensor<1x40x1536xf32>) outs(%3946 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3964 = math.fpowi %in, %c2_i32_838 : f32, i32
      linalg.yield %3964 : f32
    } -> tensor<1x40x1536xf32>
    %3948 = tosa.reduce_sum %3947 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %3949 = "tosa.const"() <{value = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3950 = tosa.reciprocal %3949 : (tensor<1xf32>) -> tensor<1xf32>
    %3951 = tosa.reshape %3950 {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %3952 = tosa.mul %3951, %3948 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3953 = "tosa.const"() <{value = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %3954 = tosa.add %3952, %3953 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3955 = tosa.rsqrt %3954 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %3956 = tosa.mul %3945, %3955 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1536xf32>
    %3957 = tosa.reshape %arg340 {new_shape = array<i64: 1, 1, 1536>} : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %3958 = tosa.mul %3957, %3956 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %extracted_slice_839 = tensor.extract_slice %3958[0, 0, 0] [1, 40, 1536] [1, 1, 1] : tensor<1x40x1536xf32> to tensor<1x40x1536xf32>
    %extracted_slice_840 = tensor.extract_slice %extracted_slice_839[0, 0, 0] [1, 40, 1536] [1, 1, 1] : tensor<1x40x1536xf32> to tensor<1x40x1536xf32>
    %extracted_slice_841 = tensor.extract_slice %extracted_slice_840[0, 0, 0] [1, 40, 1536] [1, 1, 1] : tensor<1x40x1536xf32> to tensor<1x40x1536xf32>
    // ===========================================
    // 10. FINAL OUTPUT PROJECTION LAYER
    // ===========================================
    // Project the final hidden states back to vocabulary space
    // Input: Final hidden states (1x40x1536) from last transformer layer
    // Weights: %arg341 (151936x1536) - vocabulary_size x hidden_dim
    // Output: Logits (1x40x151936) - probabilities over vocabulary

    %3959 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3960 = tosa.transpose %arg341, %3959 : (tensor<151936x1536xf32>, tensor<2xi32>) -> tensor<1536x151936xf32>
    %3961 = tosa.reshape %extracted_slice_841 {new_shape = array<i64: 40, 1536>} : (tensor<1x40x1536xf32>) -> tensor<40x1536xf32>
    %cst_842 = arith.constant dense<0.000000e+00> : tensor<40x151936xf32>
    %3962 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3961, %3960 : tensor<40x1536xf32>, tensor<1536x151936xf32>) outs(%cst_842 : tensor<40x151936xf32>) -> tensor<40x151936xf32>
    %3963 = tosa.reshape %3962 {new_shape = array<i64: 1, 40, 151936>} : (tensor<40x151936xf32>) -> tensor<1x40x151936xf32>
    return %3963 : tensor<1x40x151936xf32>
  }
}
