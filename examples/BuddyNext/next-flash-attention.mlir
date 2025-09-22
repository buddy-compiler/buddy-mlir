// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-openmp \
// RUN:     -func-bufferize-dynamic-offset \
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
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libomp%shlibext \
// RUN: | FileCheck %s

// Flash Attention MLIR Implementation - Parallel Version
// Reference Paper: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

// Highly parallelized Flash Attention variant
func.func @flash_attention(
    %Q: memref<?x?x?xf32>,
    %K: memref<?x?x?xf32>,
    %V: memref<?x?x?xf32>,
    %O: memref<?x?x?xf32>,
    %scale: f32) {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %batch_size = memref.dim %Q, %c0 : memref<?x?x?xf32>
  %seq_len = memref.dim %Q, %c1 : memref<?x?x?xf32>
  %hidden_size = memref.dim %Q, %c2 : memref<?x?x?xf32>

  // Split sequence into small blocks through block_size_q, block_size_kv
  %block_size_q = arith.constant 32 : index  // Smaller block size improves parallelism
  %block_size_kv = arith.constant 32 : index

  // Constants
  %neg_inf = arith.constant -1.0e+30 : f32
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32

  // Calculate number of blocks
  %num_q_blocks_raw = arith.ceildivsi %seq_len, %block_size_q : index
  %num_kv_blocks_raw = arith.ceildivsi %seq_len, %block_size_kv : index

  // Nested parallelization for batch and Query blocks
  // num_head dimension folded into batch dimension
  scf.parallel (%batch, %q_block_idx) = (%c0, %c0) to (%batch_size, %num_q_blocks_raw) step (%c1, %c1) {

    %q_start = arith.muli %q_block_idx, %block_size_q : index
    %q_end = arith.addi %q_start, %block_size_q : index
    %q_end_clamped = arith.minsi %q_end, %seq_len : index
    %q_block_size = arith.subi %q_end_clamped, %q_start : index

    // Allocate private storage for current Query block
    %temp_o = memref.alloc(%q_block_size, %hidden_size) : memref<?x?xf32>
    %temp_l = memref.alloc(%q_block_size) : memref<?xf32>
    %temp_m = memref.alloc(%q_block_size) : memref<?xf32>

    // Initialize
    scf.parallel (%i, %d) = (%c0, %c0) to (%q_block_size, %hidden_size) step (%c1, %c1) {
      memref.store %zero, %temp_o[%i, %d] : memref<?x?xf32>
    }
    scf.parallel (%i) = (%c0) to (%q_block_size) step (%c1) {
      memref.store %neg_inf, %temp_m[%i] : memref<?xf32>
      memref.store %zero, %temp_l[%i] : memref<?xf32>
    }

    // Key-Value block loop
    scf.for %kv_start = %c0 to %seq_len step %block_size_kv {
      %kv_end = arith.addi %kv_start, %block_size_kv : index
      %kv_end_clamped = arith.minsi %kv_end, %seq_len : index
      %kv_block_size = arith.subi %kv_end_clamped, %kv_start : index

      %scores = memref.alloc(%q_block_size, %kv_block_size) : memref<?x?xf32>

      // Highly parallel matrix multiplication
      scf.parallel (%i, %j) = (%c0, %c0) to (%q_block_size, %kv_block_size) step (%c1, %c1) {
        %score = scf.for %k = %c0 to %hidden_size step %c1
                 iter_args(%acc = %zero) -> (f32) {
          %qi = arith.addi %q_start, %i : index
          %kj = arith.addi %kv_start, %j : index
          %q_val = memref.load %Q[%batch, %qi, %k] : memref<?x?x?xf32>
          %k_val = memref.load %K[%batch, %kj, %k] : memref<?x?x?xf32>
          %prod = arith.mulf %q_val, %k_val : f32
          %new_acc = arith.addf %acc, %prod : f32
          scf.yield %new_acc : f32
        }
        %scaled_score = arith.mulf %score, %scale : f32
        memref.store %scaled_score, %scores[%i, %j] : memref<?x?xf32>
      }

      // Row-level parallel processing
      scf.parallel (%i) = (%c0) to (%q_block_size) step (%c1) {
        %old_m = memref.load %temp_m[%i] : memref<?xf32>

        // Find maximum value
        %new_m = scf.for %j = %c0 to %kv_block_size step %c1
                 iter_args(%max_val = %old_m) -> (f32) {
          %score_val = memref.load %scores[%i, %j] : memref<?x?xf32>
          %new_max = arith.maximumf %max_val, %score_val : f32
          scf.yield %new_max : f32
        }

        // Calculate exp and sum
        %exp_sum = scf.for %j = %c0 to %kv_block_size step %c1
                   iter_args(%sum = %zero) -> (f32) {
          %score_val = memref.load %scores[%i, %j] : memref<?x?xf32>
          %exp_arg = arith.subf %score_val, %new_m : f32
          %exp_val = math.exp %exp_arg : f32
          memref.store %exp_val, %scores[%i, %j] : memref<?x?xf32>
          %new_sum = arith.addf %sum, %exp_val : f32
          scf.yield %new_sum : f32
        }

        %old_l = memref.load %temp_l[%i] : memref<?xf32>
        %m_diff = arith.subf %old_m, %new_m : f32
        %old_scale = math.exp %m_diff : f32
        %scaled_old_l = arith.mulf %old_l, %old_scale : f32
        %new_l = arith.addf %exp_sum, %scaled_old_l : f32

        // Update statistics
        memref.store %new_m, %temp_m[%i] : memref<?xf32>
        memref.store %new_l, %temp_l[%i] : memref<?xf32>

        // Parallel output update
        scf.parallel (%d) = (%c0) to (%hidden_size) step (%c1) {
          %old_o = memref.load %temp_o[%i, %d] : memref<?x?xf32>
          %scaled_old_o = arith.mulf %old_o, %old_scale : f32

          %new_contrib = scf.for %j = %c0 to %kv_block_size step %c1
                        iter_args(%acc = %zero) -> (f32) {
            %vj = arith.addi %kv_start, %j : index
            %v_val = memref.load %V[%batch, %vj, %d] : memref<?x?x?xf32>
            %softmax_val = memref.load %scores[%i, %j] : memref<?x?xf32>
            %contrib = arith.mulf %v_val, %softmax_val : f32
            %new_acc = arith.addf %acc, %contrib : f32
            scf.yield %new_acc : f32
          }

          %updated_o = arith.addf %scaled_old_o, %new_contrib : f32
          memref.store %updated_o, %temp_o[%i, %d] : memref<?x?xf32>
        }
      }

      memref.dealloc %scores : memref<?x?xf32>
    }

    // Parallel normalization and write
    scf.parallel (%i, %d) = (%c0, %c0) to (%q_block_size, %hidden_size) step (%c1, %c1) {
      %final_l = memref.load %temp_l[%i] : memref<?xf32>
      %inv_l = arith.divf %one, %final_l : f32
      %o_val = memref.load %temp_o[%i, %d] : memref<?x?xf32>
      %normalized_o = arith.mulf %o_val, %inv_l : f32
      %qi = arith.addi %q_start, %i : index
      memref.store %normalized_o, %O[%batch, %qi, %d] : memref<?x?x?xf32>
    }

    // Cleanup
    memref.dealloc %temp_o : memref<?x?xf32>
    memref.dealloc %temp_l : memref<?xf32>
    memref.dealloc %temp_m : memref<?xf32>
  }

  return
}

// Q (1x1024x128)
// K (1x1024x128)
// V (1x1024x128)
// O (1x1024x128)
// scale (128)

func.func @alloc_f32(%d0: index, %d1: index, %d2: index, %val: f32) -> memref<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%d0,%d1,%d2) : memref<?x?x?xf32>
  scf.for %i = %c0 to %d0 step %c1 {
    scf.for %j = %c0 to %d1 step %c1 {
      scf.for %k = %c0 to %d2 step %c1 {
        memref.store %val, %0[%i,%j,%k] : memref<?x?x?xf32>
      }
    }
  }
  return %0 : memref<?x?x?xf32>
}

func.func @main() {
  %d0 = arith.constant 1 : index
  %d1 = arith.constant 16384 : index
  %d2 = arith.constant 128 : index

  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %x = func.call @alloc_f32(%d0, %d1, %d2, %f2) : (index, index, index, f32) -> memref<?x?x?xf32>
  %a = arith.constant 5.0 : f32

  %t_start = call @rtclock() : () -> f64
  func.call @flash_attention(%x, %x, %x, %x, %a) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>, f32) -> ()
  %t_end = call @rtclock() : () -> f64

  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  return
}

func.func private @rtclock() -> f64
