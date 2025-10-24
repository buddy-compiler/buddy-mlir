// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
//
// CHECK: {{[0-9]+\.[0-9]+}}

module {
  memref.global "private" constant @__constant_1x40x151936xf32 : memref<1x40x151936xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  
  func.func @softmax_kernel(%arg0: memref<1x40x151936xf32>) {
    %c151936 = arith.constant 151936 : index
    %c40 = arith.constant 40 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_neg_inf = arith.constant -3.40282347E+38 : f32
    
    %0 = call @rtclock() : () -> f64
    
    %result = memref.alloc() {alignment = 64 : i64} : memref<1x40x151936xf32>
    
    scf.for %row = %c0 to %c40 step %c1 {
      %max_scalar = memref.alloc() : memref<f32>
      memref.store %cst_neg_inf, %max_scalar[] : memref<f32>
      
      %remainder_max = arith.remsi %c151936, %c8 : index
      %vectorized_end_max = arith.subi %c151936, %remainder_max : index
      
      // max(x)
      %neg_inf_vec = vector.splat %cst_neg_inf : vector<8xf32>
      %max_vec = memref.alloc() : memref<vector<8xf32>>
      memref.store %neg_inf_vec, %max_vec[] : memref<vector<8xf32>>
      
      scf.for %col = %c0 to %vectorized_end_max step %c8 {
        %input_vec = vector.load %arg0[%c0, %row, %col] : memref<1x40x151936xf32>, vector<8xf32>
        %current_max = memref.load %max_vec[] : memref<vector<8xf32>>
        %new_max = arith.maximumf %input_vec, %current_max : vector<8xf32>
        memref.store %new_max, %max_vec[] : memref<vector<8xf32>>
      }
      scf.for %col = %vectorized_end_max to %c151936 step %c1 {
        %val = memref.load %arg0[%c0, %row, %col] : memref<1x40x151936xf32>
        %current_max = memref.load %max_scalar[] : memref<f32>
        %new_max = arith.maximumf %val, %current_max : f32
        memref.store %new_max, %max_scalar[] : memref<f32>
      }
      
      %final_max_vec = memref.load %max_vec[] : memref<vector<8xf32>>
      %max_scalar_from_vec = vector.reduction <maximumf>, %final_max_vec : vector<8xf32> into f32
      %current_scalar_max = memref.load %max_scalar[] : memref<f32>
      %final_max = arith.maximumf %max_scalar_from_vec, %current_scalar_max : f32
      
      // sum(exp(x-max))
      %sum_scalar = memref.alloc() : memref<f32>
      memref.store %cst, %sum_scalar[] : memref<f32>
      
      %max_vec_broadcast = vector.splat %final_max : vector<8xf32>
      %zero_vec = vector.splat %cst : vector<8xf32>
      %sum_vec = memref.alloc() : memref<vector<8xf32>>
      memref.store %zero_vec, %sum_vec[] : memref<vector<8xf32>>
      
      scf.for %col = %c0 to %vectorized_end_max step %c8 {
        %input_vec = vector.load %arg0[%c0, %row, %col] : memref<1x40x151936xf32>, vector<8xf32>
        %sub_vec = arith.subf %input_vec, %max_vec_broadcast : vector<8xf32>
        %exp_vec = math.exp %sub_vec : vector<8xf32>

        vector.store %exp_vec, %result[%c0, %row, %col] : memref<1x40x151936xf32>, vector<8xf32>
        
        %current_sum = memref.load %sum_vec[] : memref<vector<8xf32>>
        %new_sum = arith.addf %current_sum, %exp_vec : vector<8xf32>
        memref.store %new_sum, %sum_vec[] : memref<vector<8xf32>>
      }
      scf.for %col = %vectorized_end_max to %c151936 step %c1 {
        %val = memref.load %arg0[%c0, %row, %col] : memref<1x40x151936xf32>
        %sub_val = arith.subf %val, %final_max : f32
        %exp_val = math.exp %sub_val : f32
        memref.store %exp_val, %result[%c0, %row, %col] : memref<1x40x151936xf32>
        
        %current_sum = memref.load %sum_scalar[] : memref<f32>
        %new_sum = arith.addf %current_sum, %exp_val : f32
        memref.store %new_sum, %sum_scalar[] : memref<f32>
      }
      
      %final_sum_vec = memref.load %sum_vec[] : memref<vector<8xf32>>
      %sum_from_vec = vector.reduction <add>, %final_sum_vec : vector<8xf32> into f32
      %scalar_sum = memref.load %sum_scalar[] : memref<f32>
      %total_sum = arith.addf %sum_from_vec, %scalar_sum : f32
      
      // log_sum_exp = max + log(sum)
      // exp(x - log_sum_exp)
      %log_sum = math.log %total_sum : f32
      %log_sum_exp = arith.addf %final_max, %log_sum : f32
      %log_sum_exp_vec = vector.splat %log_sum_exp : vector<8xf32>
      
      scf.for %col = %c0 to %vectorized_end_max step %c8 {
        %input_vec = vector.load %arg0[%c0, %row, %col] : memref<1x40x151936xf32>, vector<8xf32>
        %sub_vec = arith.subf %input_vec, %log_sum_exp_vec : vector<8xf32>
        %softmax_vec = math.exp %sub_vec : vector<8xf32>
        vector.store %softmax_vec, %result[%c0, %row, %col] : memref<1x40x151936xf32>, vector<8xf32>
      }
      
      scf.for %col = %vectorized_end_max to %c151936 step %c1 {
        %input_val = memref.load %arg0[%c0, %row, %col] : memref<1x40x151936xf32>
        %sub_val = arith.subf %input_val, %log_sum_exp : f32
        %softmax_val = math.exp %sub_val : f32
        memref.store %softmax_val, %result[%c0, %row, %col] : memref<1x40x151936xf32>
      }
      
      memref.dealloc %max_scalar : memref<f32>
      memref.dealloc %max_vec : memref<vector<8xf32>>
      memref.dealloc %sum_scalar : memref<f32>
      memref.dealloc %sum_vec : memref<vector<8xf32>>
    }
    
    %1 = call @rtclock() : () -> f64
    %2 = arith.subf %1, %0 : f64
    %cast = memref.cast %result : memref<1x40x151936xf32> to memref<*xf32>
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    
    memref.dealloc %result : memref<1x40x151936xf32>
    vector.print %2 : f64
    return
  }
  
  func.func @main() {
    %0 = memref.get_global @__constant_1x40x151936xf32 : memref<1x40x151936xf32>
    call @softmax_kernel(%0) : (memref<1x40x151936xf32>) -> ()
    return
  }
}