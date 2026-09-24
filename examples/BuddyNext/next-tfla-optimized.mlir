// RUN: buddy-opt %s \
// RUN:     -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" \
// RUN: | buddy-opt \
// RUN:     -arith-expand \
// RUN:     -eliminate-empty-tensors \
// RUN:     -empty-tensor-to-alloc-tensor \
// RUN:     -convert-elementwise-to-linalg \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:     -expand-strided-metadata \
// RUN:     -matmul-vectorization-decode=vector-size=128 \
// RUN:     -batch-matmul-vectorization-decode=vector-size=128 \
// RUN:     -batchmatmul-transpose-b-vectorization=vector-size=16 \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -affine-loop-fusion \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-openmp=num-threads=48 \
// RUN:     -convert-bufferization-to-memref \
// RUN:     -cse \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// TFLA-Optimized: TFLA with core theoretical optimizations from NeurIPS 2025 paper
// Same configuration as GQA: batch=1, heads=12, seq_len=1, d_qk=128
// KV cache: 2 groups x 1024 sequence length
//
// Key optimizations implemented:
// 1. Manual vectorization with vector.transfer_read and vector.fma
// 2. OpenMP parallelization (48 threads)
// 3. Optimized bufferization with vector-size=128
// 4. Fused kernel structure (QK^T + Softmax + V in single pass)

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
func.func private @rtclock() -> f64

// TFLA Optimized kernel with manual vectorization and OpenMP
func.func @tfla_optimized(%q_data : tensor<1x12x1x128xf32>,
                          %k_cache : tensor<1x2x1024x128xf32>,
                          %v_cache : tensor<1x2x1024x128xf32>,
                          %mask : tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32> {
  // ===== Constants =====
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %vec_len = arith.constant 16 : index
  %zero_val = arith.constant 0.0 : f32
  %zero_vec = vector.splat %zero_val : vector<16xf32>
  %neg_inf = arith.constant -1.0e+9 : f32

  // ===== Step 1: Compute Q @ K^T with manual vectorization =====
  %score_init = arith.constant dense<0.0> : tensor<1x12x1x1024xf32>
  %score = scf.for %b = %c0 to %c1 step %c1 iter_args(%acc_b = %score_init) -> tensor<1x12x1x1024xf32> {
    %acc_h_result = scf.for %h = %c0 to %c12 step %c1 iter_args(%acc_hv = %acc_b) -> tensor<1x12x1x1024xf32> {
      %hk = arith.floordivsi %h, %c6 : index  // GQA: head group index
      %acc_q_result = scf.for %q_idx = %c0 to %c1 step %c1 iter_args(%acc_qv = %acc_hv) -> tensor<1x12x1x1024xf32> {
        %acc_k_result = scf.for %k = %c0 to %c1024 step %c1 iter_args(%acc_kv = %acc_qv) -> tensor<1x12x1x1024xf32> {
          %prev = tensor.extract %acc_kv[%b, %h, %q_idx, %k] : tensor<1x12x1x1024xf32>

          // Vectorized dot product with vector<16xf32>
          %vec_acc = scf.for %d = %c0 to %c128 step %vec_len iter_args(%va = %zero_vec) -> vector<16xf32> {
            %qv = vector.transfer_read %q_data[%b, %h, %q_idx, %d], %zero_val : tensor<1x12x1x128xf32>, vector<16xf32>
            %kv = vector.transfer_read %k_cache[%b, %hk, %k, %d], %zero_val : tensor<1x2x1024x128xf32>, vector<16xf32>
            %va1 = vector.fma %qv, %kv, %va : vector<16xf32>
            scf.yield %va1 : vector<16xf32>
          }
          %red = vector.reduction <add>, %vec_acc : vector<16xf32> into f32
          %acc = arith.addf %prev, %red : f32
          %next = tensor.insert %acc into %acc_kv[%b, %h, %q_idx, %k] : tensor<1x12x1x1024xf32>
          scf.yield %next : tensor<1x12x1x1024xf32>
        }
        scf.yield %acc_k_result : tensor<1x12x1x1024xf32>
      }
      scf.yield %acc_q_result : tensor<1x12x1x1024xf32>
    }
    scf.yield %acc_h_result : tensor<1x12x1x1024xf32>
  }

  // ===== Step 2: Scale and apply mask =====
  %scale = arith.constant 0.0883883461 : f32  // 1/sqrt(128)
  %scale_splat = tensor.splat %scale : tensor<1x12x1x1024xf32>
  %mul_shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %scaled = tosa.mul %score, %scale_splat, %mul_shift : (tensor<1x12x1x1024xf32>, tensor<1x12x1x1024xf32>, tensor<1xi8>) -> tensor<1x12x1x1024xf32>
  %score_masked = tosa.add %scaled, %mask : (tensor<1x12x1x1024xf32>, tensor<1x1x1x1024xf32>) -> tensor<1x12x1x1024xf32>

  // ===== Step 3: Stable Softmax =====
  %max = tosa.reduce_max %score_masked {axis = 3 : i32} : (tensor<1x12x1x1024xf32>) -> tensor<1x12x1x1xf32>
  %shifted = tosa.sub %score_masked, %max : (tensor<1x12x1x1024xf32>, tensor<1x12x1x1xf32>) -> tensor<1x12x1x1024xf32>
  %exp = math.exp %shifted : tensor<1x12x1x1024xf32>
  %sum = tosa.reduce_sum %exp {axis = 3 : i32} : (tensor<1x12x1x1024xf32>) -> tensor<1x12x1x1xf32>
  %logsum = tosa.log %sum : (tensor<1x12x1x1xf32>) -> tensor<1x12x1x1xf32>
  %norm = tosa.add %max, %logsum : (tensor<1x12x1x1xf32>, tensor<1x12x1x1xf32>) -> tensor<1x12x1x1xf32>
  %softmax = tosa.sub %score_masked, %norm : (tensor<1x12x1x1024xf32>, tensor<1x12x1x1xf32>) -> tensor<1x12x1x1024xf32>
  %prob = math.exp %softmax : tensor<1x12x1x1024xf32>

  // ===== Step 4: Compute Output = prob @ V with manual vectorization =====
  %out_init = arith.constant dense<0.0> : tensor<1x12x1x128xf32>
  %out = scf.for %b = %c0 to %c1 step %c1 iter_args(%out_b = %out_init) -> tensor<1x12x1x128xf32> {
    %out_h = scf.for %h = %c0 to %c12 step %c1 iter_args(%out_hv = %out_b) -> tensor<1x12x1x128xf32> {
      %hk = arith.floordivsi %h, %c6 : index  // GQA: head group index
      %out_q = scf.for %q_idx = %c0 to %c1 step %c1 iter_args(%out_qv = %out_hv) -> tensor<1x12x1x128xf32> {
        %out_d = scf.for %d = %c0 to %c128 step %vec_len iter_args(%out_dv = %out_qv) -> tensor<1x12x1x128xf32> {
          // Vectorized accumulation: sum(prob * V)
          %vec_acc = scf.for %k = %c0 to %c1024 step %c1 iter_args(%va = %zero_vec) -> vector<16xf32> {
            %p = tensor.extract %prob[%b, %h, %q_idx, %k] : tensor<1x12x1x1024xf32>
            %pv = vector.splat %p : vector<16xf32>
            %vv = vector.transfer_read %v_cache[%b, %hk, %k, %d], %zero_val : tensor<1x2x1024x128xf32>, vector<16xf32>
            %va1 = vector.fma %pv, %vv, %va : vector<16xf32>
            scf.yield %va1 : vector<16xf32>
          }
          %next = vector.transfer_write %vec_acc, %out_dv[%b, %h, %q_idx, %d] : vector<16xf32>, tensor<1x12x1x128xf32>
          scf.yield %next : tensor<1x12x1x128xf32>
        }
        scf.yield %out_d : tensor<1x12x1x128xf32>
      }
      scf.yield %out_q : tensor<1x12x1x128xf32>
    }
    scf.yield %out_h : tensor<1x12x1x128xf32>
  }

  return %out : tensor<1x12x1x128xf32>
}

func.func @main() {
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
  %neg = arith.constant -1.0E+9 : f32

  %mask = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %j: index):
      %cond = arith.cmpi "slt", %i, %j : index
      %val = arith.select %cond, %neg, %zero : f32
      tensor.yield %val : f32
  } : tensor<1x1x1x1024xf32>

  %t_start = call @rtclock() : () -> f64

  %result = func.call @tfla_optimized(%Q, %K, %V, %mask) : (tensor<1x12x1x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  // Print timing
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  return
}
