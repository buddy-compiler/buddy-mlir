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
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s



#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0) -> (d0)>
func.func private @printMemrefF32(%ptr : tensor<*xf32>)
func.func private @rtclock() -> f64
func.func @kernel(%q : tensor<1x12x1x128xf32>, %k_cache : tensor<1x2x1024x128xf32>, %v_cache : tensor<1x2x1024x128xf32>, %mask : tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32> {

  %c0_21 = arith.constant 0 : index
  %cst_22 = arith.constant 1.000000e+00 : f32
  %cst_23 = arith.constant 0.0883883461 : f32
  %cst_24 = arith.constant 0.000000e+00 : f32
  %cst_25 = arith.constant -1.000000e+30 : f32
  %q_mem = bufferization.to_buffer %q : tensor<1x12x1x128xf32> to memref<1x12x1x128xf32>
  %k_cache_mem = bufferization.to_buffer %k_cache : tensor<1x2x1024x128xf32> to memref<1x2x1024x128xf32>
  %v_cache_mem = bufferization.to_buffer %v_cache : tensor<1x2x1024x128xf32> to memref<1x2x1024x128xf32>
  %mask_mem = bufferization.to_buffer %mask : tensor<1x1x1x1024xf32> to memref<1x1x1x1024xf32>
  %batch = arith.constant 1 : index
  %head_num = arith.constant 12 : index
  %seq_len = arith.constant 1 : index
  %head_dim = arith.constant 128 : index
  %out_mem = memref.alloc() : memref<1x12x1x128xf32>
  %value_mem = memref.alloc() : memref<1x12x1xf32>
  %accum_mem = memref.alloc() : memref<128xf32>

  %vec_len = arith.constant 16 : index
  %vec_f32 = vector.splat %cst_24 : vector<16xf32>

  affine.for %b = 0 to #map2(%batch) {
    affine.for %h = 0 to #map2(%head_num) {
      %c6 = arith.constant 6 : index
      %h_kv = arith.divsi %h, %c6 : index
      affine.for %i = 0 to #map2(%seq_len) {
        affine.for %k= 0 to #map2(%head_dim) {
          %cst_1253 = arith.constant 0.000000e+00 : f32
          memref.store %cst_1253, %accum_mem[%k] : memref<128xf32>
        }
        %k_seq_len = arith.constant 1024 : index
        %2738:2 = affine.for %k= 0 to #map2(%k_seq_len) iter_args(%m_temp = %cst_25, %l_temp = %cst_24) -> (f32, f32) {
          %acc = affine.for %k1 = 0 to #map2(%head_dim) step 16 iter_args(%acc = %vec_f32) -> (vector<16xf32>) {
            %q_data = vector.load %q_mem[%b, %h, %i, %k1] : memref<1x12x1x128xf32>, vector<16xf32>
            %k_data = vector.load %k_cache_mem[%b, %h_kv, %k, %k1] : memref<1x2x1024x128xf32>, vector<16xf32>
            %new_acc = vector.fma %q_data, %k_data, %acc : vector<16xf32>
            affine.yield %new_acc : vector<16xf32>
          }
          %score_sum = vector.reduction <add>, %acc : vector<16xf32> into f32
          %score_scaled = arith.mulf %score_sum, %cst_23 : f32
          %mask_val = affine.load %mask_mem[%b, %c0_21, %i, %k] : memref<1x1x1x1024xf32>
          %score_masked = arith.addf %score_scaled, %mask_val : f32
          %cond_max = arith.cmpf ogt, %score_masked, %m_temp : f32
          %new_max = arith.select %cond_max, %score_masked, %m_temp : f32
          %sub1 = arith.subf %m_temp, %score_masked : f32
          %exp1 = math.exp %sub1 : f32
          %mul1 = arith.mulf %exp1, %l_temp : f32
          %add1 = arith.addf %mul1, %cst_22 : f32

          %sub2 = arith.subf %score_masked, %m_temp : f32
          %exp2 = math.exp %sub2 : f32
          %add2 = arith.addf %l_temp, %exp2 : f32

          %sum_exp_update = arith.select %cond_max, %add1, %add2 : f32
          affine.for %k1 = 0 to #map2(%head_dim) step 16{
            %v_data = memref.load %v_cache_mem[%b, %h_kv, %k, %k1] : memref<1x2x1024x128xf32>
            %acc_old = memref.load %accum_mem[%k1] : memref<128xf32>
            %acc_mul1 = arith.mulf %acc_old, %exp1 : f32
            %r1 = arith.addf %acc_mul1, %v_data : f32
            %acc_mul2 = arith.mulf %exp2, %v_data : f32
            %r2 = arith.addf %acc_mul2, %acc_old : f32
            %acc_new = arith.select %cond_max, %r1, %r2 : f32
            memref.store %acc_new, %accum_mem[%k1] : memref<128xf32>
          }
          affine.yield %new_max, %sum_exp_update : f32, f32
        }
        memref.store %2738#1, %value_mem[%b, %h, %i] : memref<1x12x1xf32>
        affine.for %k= 0 to #map2(%head_dim) step 16{
          %accum_temp = memref.load %accum_mem[%k] : memref<128xf32>
          %accum_temp_div = arith.divf %accum_temp, %2738#1 : f32
          memref.store %accum_temp_div, %out_mem[%b, %h, %i, %k] : memref<1x12x1x128xf32>
        }
      }
    }
  }
  %result = bufferization.to_tensor %out_mem restrict : memref<1x12x1x128xf32> to tensor<1x12x1x128xf32>

    return %result : tensor<1x12x1x128xf32>
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


    // Mask: only allow j <= i positions, simulate causal mask
    %zero = arith.constant 0.0 : f32
    %neg  = arith.constant -1.0E+9 : f32

    %mask = tensor.generate {
    ^bb0(%b: index, %h: index, %i: index, %j: index):
      %cond = arith.cmpi "slt", %i, %j : index
      %val = arith.select %cond, %neg, %zero : f32
      tensor.yield %val : f32
    } : tensor<1x1x1x1024xf32>

  %t_start = call @rtclock() : () -> f64

  %result_out = call @kernel(%Q, %K, %V, %mask) : (tensor<1x12x1x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x2x1024x128xf32>, tensor<1x1x1x1024xf32>) -> tensor<1x12x1x128xf32>

  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  vector.print %time : f64
  // CHECK: {{[0-9]+\.[0-9]+}}

  %tensor_unranked = tensor.cast %result_out : tensor<1x12x1x128xf32> to tensor<*xf32>
  // call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()

  return
}
