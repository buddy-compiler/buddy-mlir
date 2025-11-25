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

// Flash Attention MLIR Implementation - vectorized + tiled version
// Reference Paper: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
// Vectorized + tiled version of Flash Attention with head tiling

func.func private @rtclock() -> f64
func.func @kernel(
  %Q: tensor<1x12x1x128xf32>,
  %K: tensor<1x12x1024x128xf32>,
  %V: tensor<1x12x1024x128xf32>,
  %attention_mask: tensor<1x1x1x1024xf32>
) -> (tensor<?x?x?x?xf32>, tensor<?x?x?xf32>) {

  %c0 = arith.constant 0 : index
  %vec_len = arith.constant 16 : index
  %step_1 = arith.constant 1 : index
  %one_f32 = arith.constant 1.000000e+00 : f32
  %scale = arith.constant 0.0883883461 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %zero_vec = vector.splat %zero : vector<16xf32>
  %cst_25 = arith.constant -1.000000e+30 : f32
  %Q_memref = bufferization.to_memref %Q : tensor<1x12x1x128xf32> to memref<1x12x1x128xf32>
  %K_memref = bufferization.to_memref %K : tensor<1x12x1024x128xf32> to memref<1x12x1024x128xf32>
  %V_memref = bufferization.to_memref %V : tensor<1x12x1024x128xf32> to memref<1x12x1024x128xf32>
  %attention_mask_memref = bufferization.to_memref %attention_mask : tensor<1x1x1x1024xf32> to memref<1x1x1x1024xf32>
  %batch_size = arith.constant 1 : index
  %num_heads = arith.constant 12 : index
  %q_seq_len = arith.constant 1 : index
  %k_seq_len = arith.constant 1024 : index
  %head_dim = arith.constant 128 : index

  %out_scores = memref.alloc(%batch_size,%num_heads,%q_seq_len) : memref<?x?x?xf32>
  %out = memref.alloc(%batch_size,%num_heads,%q_seq_len,%head_dim) : memref<?x?x?x?xf32>

  %block_size_kv = arith.constant 32 : index
  %block_size_head = arith.constant 4 : index



  scf.for %b = %c0 to %batch_size step %step_1 {
    scf.for %q = %c0 to %q_seq_len step %step_1 {
      scf.for %h_block_start = %c0 to %num_heads step %block_size_head {
        %m_i_memref = memref.alloc(%block_size_head) : memref<?xf32>
        %l_i_memref = memref.alloc(%block_size_head) : memref<?xf32>
        %accm_memref = memref.alloc(%block_size_head,%head_dim) : memref<?x?xf32>
        // initialize m_i, l_i
        scf.for %jj = %c0 to %block_size_head step %step_1 {
          memref.store %zero, %m_i_memref[%jj] : memref<?xf32>
          memref.store %zero, %l_i_memref[%jj] : memref<?xf32>
        }
        scf.for %jj = %c0 to %block_size_head step %step_1 {
          scf.for %k = %c0 to %head_dim step %vec_len {
            vector.store %zero_vec, %accm_memref[%jj, %k] : memref<?x?xf32>, vector<16xf32>
          }
        }
        // kv block iteration
        scf.for %k_block_start = %c0 to %block_size_kv step %step_1 {
          scf.for %h_local = %c0 to %block_size_head step %step_1 {
            %idx_h = arith.addi %h_block_start, %h_local : index
            %score_tile_memref = memref.alloc(%block_size_kv) : memref<?xf32>
            %acc_block_memref = memref.alloc(%head_dim) : memref<?xf32>
            //initialize score_tile
            scf.for %jj = %c0 to %block_size_kv step %vec_len {
              vector.store %zero_vec, %score_tile_memref[%jj] : memref<?xf32>, vector<16xf32>
              }
              //compute score_tile
            %m_block = scf.for %jj = %c0 to %block_size_kv step %step_1 iter_args(%max_block_iter = %cst_25) -> (f32) {
              %idx_kv = arith.addi %k_block_start, %jj : index
              %acc = scf.for %k = %c0 to %head_dim step %vec_len  iter_args(%acc = %zero_vec) -> (vector<16xf32>) {
                %q_data = vector.transfer_read %Q_memref[%b, %idx_h, %q, %k], %zero {in_bounds = [true]} : memref<1x12x1x128xf32>, vector<16xf32>
                %k_data = vector.transfer_read %K_memref[%b, %idx_h, %idx_kv, %k], %zero {in_bounds = [true]} : memref<1x12x1024x128xf32>, vector<16xf32>
                %new_acc = vector.fma %q_data, %k_data, %acc : vector<16xf32>
                scf.yield %new_acc : vector<16xf32>
              }
              %score_tile_sum = vector.reduction <add>, %acc : vector<16xf32> into f32
              %score_tile_scaled = arith.mulf %score_tile_sum, %scale : f32
              %mask_val = memref.load %attention_mask_memref[%b, %c0, %q, %idx_kv] : memref<1x1x1x1024xf32>
              %score_tile_masked = arith.addf %score_tile_scaled, %mask_val : f32
              memref.store %score_tile_masked, %score_tile_memref[%jj] : memref<?xf32>
              %is_m_i = arith.cmpf ogt, %score_tile_masked, %max_block_iter : f32
              %m_i_tile = arith.select %is_m_i, %score_tile_masked, %max_block_iter : f32
              scf.yield %m_i_tile : f32
            }
            scf.for %k = %c0 to %head_dim step %vec_len {
              vector.store %zero_vec, %acc_block_memref[%k] : memref<?xf32>, vector<16xf32>
            }
            %l_block = scf.for %jj = %c0 to %block_size_kv step %step_1 iter_args(%l_block_iter = %zero) -> (f32) {
              %idx_kv = arith.addi %k_block_start, %jj : index
              %score_tile_masked = memref.load %score_tile_memref[%jj] : memref<?xf32>
              %score_tile_sub_m_block = arith.subf %score_tile_masked, %m_block : f32
              %p = math.exp %score_tile_sub_m_block : f32
              %exp_score_tile_vec = vector.splat %p : vector<16xf32>
              %l_block_new = arith.addf %l_block_iter, %p : f32
              scf.for %k = %c0 to %head_dim step %vec_len {
                %v_data = vector.transfer_read %V_memref[%b, %idx_h, %idx_kv, %k], %zero {in_bounds = [true]} : memref<1x12x1024x128xf32>, vector<16xf32>
                %acc_block_val = vector.load %acc_block_memref[%k] : memref<?xf32>, vector<16xf32>
                %new_acc = vector.fma %v_data, %exp_score_tile_vec, %acc_block_val : vector<16xf32>
                vector.store %new_acc, %acc_block_memref[%k] : memref<?xf32>, vector<16xf32>
              }
              scf.yield %l_block_new : f32
            }
            %m_i_iter = memref.load %m_i_memref[%h_local] : memref<?xf32>
            %m_i_iter_is_max = arith.cmpf ogt, %m_block, %m_i_iter : f32
            %m_new = arith.select %m_i_iter_is_max, %m_block, %m_i_iter : f32

            %sub_max = arith.subf %m_i_iter, %m_new : f32
            %alpha = math.exp %sub_max : f32
            %alpha_vec = vector.splat %alpha : vector<16xf32>

            %sub_block = arith.subf %m_block, %m_new : f32
            %beta = math.exp %sub_block : f32
            %beta_vec = vector.splat %beta : vector<16xf32>

            scf.for %k = %c0 to %head_dim step %vec_len {
              %acc_vec = vector.load %accm_memref[%h_local, %k] : memref<?x?xf32>, vector<16xf32>
              %acc_block_vec = vector.load %acc_block_memref[%k] : memref<?xf32>, vector<16xf32>
              %alpha_mul = arith.mulf %acc_vec, %alpha_vec : vector<16xf32>
              %beta_mul = arith.mulf %acc_block_vec, %beta_vec : vector<16xf32>
              %new_acc = arith.addf %alpha_mul, %beta_mul : vector<16xf32>
              vector.store %new_acc, %accm_memref[%h_local, %k] : memref<?x?xf32>, vector<16xf32>
            }
            %l_i_iter = memref.load %l_i_memref[%h_local] : memref<?xf32>
            %l_alpha = arith.mulf %l_i_iter, %alpha : f32
            %l_beta = arith.mulf %l_block, %beta : f32
            %l_i = arith.addf %l_alpha, %l_beta : f32
            memref.store %l_i, %l_i_memref[%h_local] : memref<?xf32>
            memref.store %m_new, %m_i_memref[%h_local] : memref<?xf32>
          }
        }
        scf.for %h_local = %c0 to %block_size_head step %step_1 {
          %idx_h = arith.addi %h_block_start, %h_local : index
          %sum = memref.load %l_i_memref[%h_local] : memref<?xf32>
          %sum_vec = vector.splat %sum : vector<16xf32>
          memref.store %sum, %out_scores[%b, %idx_h, %q] : memref<?x?x?xf32>
          scf.for %k = %c0 to %head_dim step %vec_len {
            %acc_vec = vector.load %accm_memref[%h_local, %k] : memref<?x?xf32>, vector<16xf32>
            %out_vec = arith.divf %acc_vec, %sum_vec : vector<16xf32>
            vector.store %out_vec, %out[%b, %idx_h, %q, %k] : memref<?x?x?x?xf32>, vector<16xf32>
          }
        }
      }
    }
  }
  %out_tensor = bufferization.to_tensor %out restrict : memref<?x?x?x?xf32> to tensor<?x?x?x?xf32>
  %out_scores_tensor = bufferization.to_tensor %out_scores restrict : memref<?x?x?xf32> to tensor<?x?x?xf32>
  return %out_tensor, %out_scores_tensor : tensor<?x?x?x?xf32>, tensor<?x?x?xf32>
}

func.func @main() {
  %t_start = call @rtclock() : () -> f64

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
  } : tensor<1x12x1024x128xf32>

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
  } : tensor<1x12x1024x128xf32>


    // Mask: only allow j <= i positions, simulate causal mask
    %zero = arith.constant 0.0 : f32
    %neg  = arith.constant -1.0E+9 : f32

    %mask = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %j: index):
      %cond = arith.cmpi "slt", %i, %j : index
      %val = arith.select %cond, %neg, %zero : f32
      tensor.yield %val : f32
    } : tensor<1x1x1x1024xf32>
  %result_out, %result_scores = call @kernel(%Q, %K, %V, %mask) : (tensor<1x12x1x128xf32>, tensor<1x12x1024x128xf32>, tensor<1x12x1024x128xf32>, tensor<1x1x1x1024xf32>) -> (tensor<?x?x?x?xf32>, tensor<?x?x?xf32>)

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64

  %tensor_unranked = tensor.cast %result_out : tensor<?x?x?x?xf32> to tensor<*xf32>
  call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  %tensor_unranked_scores = tensor.cast %result_scores : tensor<?x?x?xf32> to tensor<?x?x?xf32>
  call @printMemrefF32(%tensor_unranked_scores) : (tensor<*xf32>) -> ()

  vector.print %time : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
  return
}
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
