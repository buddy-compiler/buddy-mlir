// RUN: buddy-opt %s \
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
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-libm  \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts | \
// RUN:     mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libomp%shlibext \
// RUN:     | FileCheck %s

// Flash Attention MLIR Implementation - vectorized version
// Reference Paper: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
// Vectorized version of Flash Attention

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0) -> (d0)>
func.func private @rtclock() -> f64
func.func @kernel(
  %Q: tensor<1x12x40x128xf32>,
  %K: tensor<1x12x40x128xf32>,
  %V: tensor<1x12x40x128xf32>,
  %Mask: tensor<1x1x40x40xf32>
) -> tensor<1x12x40x128xf32> {
  %c0 = arith.constant 0 : index
  %one_f32 = arith.constant 1.000000e+00 : f32
  %scale = arith.constant 0.0883883461 : f32
  %zero_f32 = arith.constant 0.000000e+00 : f32
  %neg_inf = arith.constant -1.000000e+30 : f32

  %Q_mem = bufferization.to_memref %Q : tensor<1x12x40x128xf32> to memref<1x12x40x128xf32>
  %K_mem = bufferization.to_memref %K : tensor<1x12x40x128xf32> to memref<1x12x40x128xf32>
  %V_mem = bufferization.to_memref %V : tensor<1x12x40x128xf32> to memref<1x12x40x128xf32>
  %Mask_mem = bufferization.to_memref %Mask : tensor<1x1x40x40xf32> to memref<1x1x40x40xf32>

  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c40 = arith.constant 40 : index
  %c128 = arith.constant 128 : index

  %output_mem = memref.alloc() : memref<1x12x40x128xf32>
  %sum_exp_mem = memref.alloc() : memref<1x12x40xf32>
  %attn_accum_mem = memref.alloc() : memref<128xf32>

  affine.for %batch = 0 to #map5(%c1) {
    affine.for %head = 0 to #map5(%c12) {
      affine.for %query_idx = 0 to #map5(%c40) {
        affine.for %dim_offset = 0 to #map5(%c128) step 16 {
          %vec_zero = vector.splat %zero_f32 : vector<16xf32>
          vector.transfer_write %vec_zero, %attn_accum_mem[%dim_offset] {in_bounds = [true]} : vector<16xf32>, memref<128xf32>
        }

        %softmax_state:2 = affine.for %key_idx = 0 to #map5(%c40) iter_args(%running_max = %neg_inf, %running_sum = %zero_f32) -> (f32, f32) {
          %qk_acc = affine.for %dim_offset = 0 to #map5(%c128) step 16 iter_args(%qk_sum = %zero_f32) -> (f32) {
            %q_vec = vector.transfer_read %Q_mem[%batch, %head, %query_idx, %dim_offset], %zero_f32 {in_bounds = [true]} : memref<1x12x40x128xf32>, vector<16xf32>
            %k_vec = vector.transfer_read %K_mem[%batch, %head, %key_idx, %dim_offset], %zero_f32 {in_bounds = [true]} : memref<1x12x40x128xf32>, vector<16xf32>
            %qk_mul = arith.mulf %q_vec, %k_vec : vector<16xf32>
            %qk_red = vector.reduction <add>, %qk_mul : vector<16xf32> into f32
            %qk_sum_next = arith.addf %qk_sum, %qk_red : f32
            affine.yield %qk_sum_next : f32
          }

          %scaled_score = arith.mulf %qk_acc, %scale : f32
          %mask_val = affine.load %Mask_mem[%batch, %c0, %query_idx, %key_idx] : memref<1x1x40x40xf32>
          %score = arith.addf %scaled_score, %mask_val : f32

          %is_new_max = arith.cmpf ogt, %score, %running_max : f32
          %new_max = arith.select %is_new_max, %score, %running_max : f32

          %diff_old = arith.subf %running_max, %score : f32
          %exp_old = math.exp %diff_old : f32
          %exp_old_running_sum = arith.mulf %exp_old, %running_sum : f32
          %sum_if_new = arith.addf %exp_old_running_sum, %one_f32 : f32

          %diff_new = arith.subf %score, %running_max : f32
          %exp_new = math.exp %diff_new : f32
          %sum_if_old = arith.addf %running_sum, %exp_new : f32

          %new_sum = arith.select %is_new_max, %sum_if_new, %sum_if_old : f32

          affine.for %dim_offset = 0 to #map5(%c128) step 16 {
            %v_vec = vector.transfer_read %V_mem[%batch, %head, %key_idx, %dim_offset], %zero_f32 {in_bounds = [true]} : memref<1x12x40x128xf32>, vector<16xf32>
            %acc_vec = vector.transfer_read %attn_accum_mem[%dim_offset], %zero_f32 {in_bounds = [true]} : memref<128xf32>, vector<16xf32>

            %exp_old_vec = vector.splat %exp_old : vector<16xf32>
            %acc_vec_exp_old_vec = arith.mulf %acc_vec, %exp_old_vec : vector<16xf32>
            %updated_if_new = arith.addf %acc_vec_exp_old_vec, %v_vec : vector<16xf32>

            %exp_new_vec = vector.splat %exp_new : vector<16xf32>
            %exp_new_vec_v_vec = arith.mulf %exp_new_vec, %v_vec : vector<16xf32>
            %updated_if_old = arith.addf %exp_new_vec_v_vec, %acc_vec : vector<16xf32>

            %acc_next = arith.select %is_new_max, %updated_if_new, %updated_if_old : vector<16xf32>
            vector.transfer_write %acc_next, %attn_accum_mem[%dim_offset] {in_bounds = [true]} : vector<16xf32>, memref<128xf32>
          }

          affine.yield %new_max, %new_sum : f32, f32
        }

        affine.store %softmax_state#1, %sum_exp_mem[%batch, %head, %query_idx] : memref<1x12x40xf32>

        affine.for %dim_offset = 0 to #map5(%c128) step 16 {
          %acc_vec = vector.transfer_read %attn_accum_mem[%dim_offset], %zero_f32 {in_bounds = [true]} : memref<128xf32>, vector<16xf32>
          %sum_vec = vector.splat %softmax_state#1 : vector<16xf32>
          %out_vec = arith.divf %acc_vec, %sum_vec : vector<16xf32>
          vector.transfer_write %out_vec, %output_mem[%batch, %head, %query_idx, %dim_offset] {in_bounds = [true]} : vector<16xf32>, memref<1x12x40x128xf32>
        }
      }
    }
  }

  %out_tensor = bufferization.to_tensor %output_mem restrict : memref<1x12x40x128xf32> to tensor<1x12x40x128xf32>
  %sum_tensor = bufferization.to_tensor %sum_exp_mem restrict : memref<1x12x40xf32> to tensor<1x12x40xf32>
  return %out_tensor : tensor<1x12x40x128xf32>
}


func.func @main() {

  %t_start = call @rtclock() : () -> f64

  %Q_val = arith.constant 3.0 : f32
  %K_val = arith.constant 2.0 : f32
  %V_val = arith.constant 8.0 : f32
  %mask_val = arith.constant 0.0 : f32  // attention mask value (0.0 means no masking)

  %Q = tensor.splat %Q_val : tensor<1x12x40x128xf32>

  %K = tensor.splat %K_val : tensor<1x12x40x128xf32>

  %V = tensor.splat %V_val : tensor<1x12x40x128xf32>

  %Mask = tensor.splat %mask_val : tensor<1x1x40x40xf32>

  %flash_result = call @kernel(%Q, %K, %V, %Mask) : (tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>, tensor<1x12x40x128xf32>, tensor<1x1x40x40xf32>) -> tensor<1x12x40x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %flash_result : tensor<1x12x40x128xf32> to tensor<*xf32>


  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  // Print timings.
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  return
}
  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
