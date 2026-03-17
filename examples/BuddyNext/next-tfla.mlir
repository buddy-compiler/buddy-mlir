// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -convert-elementwise-to-linalg \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -expand-strided-metadata \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-bufferization-to-memref \
// RUN:     -cse \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// TFLA-GQA: Fair comparison with GQA Attention (single token inference)
// Same configuration as next-gqa-attention.mlir:
//   batch=1, heads=12, seq_len=1, d_qk=128
//   KV cache: 2 groups x 1024 sequence length

func.func private @rtclock() -> f64
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

// TFLA kernel with GQA configuration (seq_len=1, 12 heads, 2 KV groups)
func.func @tfla_gqa(%q: tensor<1x12x1x128xf32>,
                    %k_cache: tensor<1x2x1024x128xf32>,
                    %v_cache: tensor<1x2x1024x128xf32>,
                    %mask: tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c0_f32 = arith.constant 0.0 : f32
  %c1_f32 = arith.constant 1.0 : f32
  %neg_inf = arith.constant -1.0e+9 : f32
  %scale_val = arith.constant 0.0883883461 : f32  // 1/sqrt(128)
  %vec_len = arith.constant 16 : index
  %zero_vec = vector.splat %c0_f32 : vector<16xf32>
  %scale_splat = vector.splat %scale_val : vector<16xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>

  // ============ Step 1: Expand GQA KV cache (2 groups -> 12 heads) ============
  // Each group serves 6 heads (12 / 2 = 6)
  %k_expanded = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %d: index):
      %group_idx = arith.divui %h, %c6 : index
      %val = tensor.extract %k_cache[%b, %group_idx, %i, %d] : tensor<1x2x1024x128xf32>
      tensor.yield %val : f32
  } : tensor<1x12x1024x128xf32>

  %v_expanded = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %d: index):
      %group_idx = arith.divui %h, %c6 : index
      %val = tensor.extract %v_cache[%b, %group_idx, %i, %d] : tensor<1x2x1024x128xf32>
      tensor.yield %val : f32
  } : tensor<1x12x1024x128xf32>

  // ============ Step 2: Reshape Q for batch matmul [1,12,1,128] -> [12,1,128] ============
  %q_reshaped = tensor.generate {
    ^bb0(%h: index, %i: index, %d: index):
      %val = tensor.extract %q[%c0, %h, %i, %d] : tensor<1x12x1x128xf32>
      tensor.yield %val : f32
  } : tensor<12x1x128xf32>

  // ============ Step 3: Reshape K^T for batch matmul [1,12,1024,128] -> [12,128,1024] ============
  %k_transposed = tensor.generate {
    ^bb0(%h: index, %d: index, %j: index):
      %val = tensor.extract %k_expanded[%c0, %h, %j, %d] : tensor<1x12x1024x128xf32>
      tensor.yield %val : f32
  } : tensor<12x128x1024xf32>

  // ============ Step 4: Compute Q @ K^T using batch_matmul ============
  %scores_init = tensor.splat %c0_f32 : tensor<12x1x1024xf32>
  %scores_raw = linalg.batch_matmul ins(%q_reshaped, %k_transposed : tensor<12x1x128xf32>, tensor<12x128x1024xf32>) outs(%scores_init : tensor<12x1x1024xf32>) -> tensor<12x1x1024xf32>

  // ============ Step 5: Scale scores by 1/sqrt(128) ============
  %scores_scaled = tensor.generate {
    ^bb0(%h: index, %i: index, %j: index):
      %val = tensor.extract %scores_raw[%h, %i, %j] : tensor<12x1x1024xf32>
      %scaled = arith.mulf %val, %scale_val : f32
      tensor.yield %scaled : f32
  } : tensor<12x1x1024xf32>

  // ============ Step 6: Broadcast mask from [1,1,1,1024] to [12,1,1024] ============
  %mask_broadcast = tensor.generate {
    ^bb0(%h: index, %i: index, %j: index):
      %val = tensor.extract %mask[%c0, %c0, %c0, %j] : tensor<1x1x1x1024xf32>
      tensor.yield %val : f32
  } : tensor<12x1x1024xf32>

  // ============ Step 7: Apply mask ============
  %scores_masked = arith.addf %scores_scaled, %mask_broadcast : tensor<12x1x1024xf32>

  // ============ Step 8: Softmax along sequence dimension ============
  %max_score = tosa.reduce_max %scores_masked {axis = 2 : i32} : (tensor<12x1x1024xf32>) -> tensor<12x1x1xf32>
  %scores_shifted = tosa.sub %scores_masked, %max_score : (tensor<12x1x1024xf32>, tensor<12x1x1xf32>) -> tensor<12x1x1024xf32>
  %scores_exp = math.exp %scores_shifted : tensor<12x1x1024xf32>
  %sum_exp = tosa.reduce_sum %scores_exp {axis = 2 : i32} : (tensor<12x1x1024xf32>) -> tensor<12x1x1xf32>
  %recip_sum = tosa.reciprocal %sum_exp : (tensor<12x1x1xf32>) -> tensor<12x1x1xf32>
  %attn_weights = tosa.mul %scores_exp, %recip_sum, %shift : (tensor<12x1x1024xf32>, tensor<12x1x1xf32>, tensor<1xi8>) -> tensor<12x1x1024xf32>

  // ============ Step 9: Reshape V for batch matmul [1,12,1024,128] -> [12,1024,128] ============
  %v_reshaped = tensor.generate {
    ^bb0(%h: index, %j: index, %d: index):
      %val = tensor.extract %v_expanded[%c0, %h, %j, %d] : tensor<1x12x1024x128xf32>
      tensor.yield %val : f32
  } : tensor<12x1024x128xf32>

  // ============ Step 10: Compute output = attn_weights @ V ============
  %output_init = tensor.splat %c0_f32 : tensor<12x1x128xf32>
  %output = linalg.batch_matmul ins(%attn_weights, %v_reshaped : tensor<12x1x1024xf32>, tensor<12x1024x128xf32>) outs(%output_init : tensor<12x1x128xf32>) -> tensor<12x1x128xf32>

  // ============ Step 11: Reshape output back to [1,12,1,128] ============
  %output_reshaped = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %d: index):
      %val = tensor.extract %output[%h, %i, %d] : tensor<12x1x128xf32>
      tensor.yield %val : f32
  } : tensor<1x12x1x128xf32>

  return %output_reshaped : tensor<1x12x1x128xf32>
}

func.func @main() {
  // %t_start = call @rtclock() : () -> f64

  // Same input generation as GQA
  %Q = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %k: index):
      %sum = arith.addi %b, %h : index
      %mix1 = arith.addi %sum, %i : index
      %mix2 = arith.addi %mix1, %k : index
      %c11 = arith.constant 11 : index
      %mod = arith.remui %mix2, %c11 : index
      %val = arith.index_cast %mod : index to i32
      %valf = arith.sitofp %val : i32 to f32
      tensor.yield %valf : f32
  } : tensor<1x12x1x128xf32>

  %K = tensor.generate {
    ^bb0(%b: index, %h: index, %k: index, %j: index):
      %sum = arith.addi %b, %h : index
      %mix1 = arith.addi %sum, %k : index
      %mix2 = arith.addi %mix1, %j : index
      %c17 = arith.constant 17 : index
      %mod = arith.remui %mix2, %c17 : index
      %val = arith.index_cast %mod : index to i32
      %valf = arith.sitofp %val : i32 to f32
      tensor.yield %valf : f32
  } : tensor<1x2x1024x128xf32>

  %V = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %k: index):
      %sum = arith.addi %b, %h : index
      %mix1 = arith.addi %sum, %i : index
      %mix2 = arith.addi %mix1, %k : index
      %c13 = arith.constant 13 : index
      %mod = arith.remui %mix2, %c13 : index
      %val = arith.index_cast %mod : index to i32
      %valf = arith.sitofp %val : i32 to f32
      tensor.yield %valf : f32
  } : tensor<1x2x1024x128xf32>

  %zero = arith.constant 0.0 : f32
  %neg  = arith.constant -1.0E+9 : f32

  %mask = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %j: index):
      %cond = arith.cmpi "slt", %i, %j : index
      %val = arith.select %cond, %neg, %zero : f32
      tensor.yield %val : f32
    } : tensor<1x1x1x1024xf32>

  %t_start = call @rtclock() : () -> f64

  %result = func.call @tfla_gqa(%Q, %K, %V, %mask) : (tensor<1x12x1x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // Don't print output (same as GQA)
  vector.print %time : f64

  return
}
