// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module attributes {
  llvm.target_triple = "x86_64-unknown-linux-gnu",
  llvm.target_cpu = "skylake-avx512",
  llvm.target_features = "+avx512f,+avx512vl,+avx512dq,+avx512bw,+fma" 
} {
  memref.global "private" constant @__constant_1x40x151936xf32 : memref<1x40x151936xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @softmax_kernel(%arg0: memref<1x40x151936xf32>) {
    %c151936 = arith.constant 151936 : index
    %c40 = arith.constant 40 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    
    %0 = call @rtclock() : () -> f64
    
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x40xf32>
    
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        memref.store %cst_0, %alloc[%arg1, %arg2] : memref<1x40xf32>
      }
    }
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        scf.for %arg3 = %c0 to %c151936 step %c1 {
          %3 = memref.load %arg0[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
          %4 = memref.load %alloc[%arg1, %arg2] : memref<1x40xf32>
          %5 = arith.cmpf ugt, %3, %4 : f32
          %6 = arith.select %5, %3, %4 : f32
          %7 = arith.cmpf uno, %4, %4 : f32
          %8 = arith.select %7, %4, %6 : f32
          memref.store %8, %alloc[%arg1, %arg2] : memref<1x40xf32>
        }
      }
    }
    %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 40, 1], strides: [40, 1, 1] : memref<1x40xf32> to memref<1x40x1xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x40x151936xf32>
    
    %c16 = arith.constant 16 : index  
    %cst_zero = arith.constant 0.000000e+00 : f32
    %zero_vector = vector.splat %cst_zero : vector<16xf32>

    %remainder = arith.remsi %c151936, %c16 : index
    %vectorized_end = arith.subi %c151936, %remainder : index
    
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        %max_scalar = memref.load %reinterpret_cast[%c0, %arg2, %c0] : memref<1x40x1xf32>
        %max_vector = vector.splat %max_scalar : vector<16xf32>
        scf.for %arg3 = %c0 to %vectorized_end step %c16 {
          %input_vector = vector.load %arg0[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>, vector<16xf32>
          %result_vector = arith.subf %input_vector, %max_vector : vector<16xf32>
          vector.store %result_vector, %alloc_1[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>, vector<16xf32>
        }
        scf.for %arg3 = %vectorized_end to %c151936 step %c1 {
          %scalar_input = memref.load %arg0[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
          %scalar_result = arith.subf %scalar_input, %max_scalar : f32
          memref.store %scalar_result, %alloc_1[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x40x151936xf32>
    
    %remainder_exp = arith.remsi %c151936, %c16 : index
    %vectorized_end_exp = arith.subi %c151936, %remainder_exp : index
    
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        scf.for %arg3 = %c0 to %vectorized_end_exp step %c16 {
          %input_vector_exp = vector.load %alloc_1[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>, vector<16xf32>
          %result_vector_exp = math.exp %input_vector_exp : vector<16xf32>
          vector.store %result_vector_exp, %alloc_2[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>, vector<16xf32>
        }
        scf.for %arg3 = %vectorized_end_exp to %c151936 step %c1 {
          %scalar_input_exp = memref.load %alloc_1[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
          %scalar_result_exp = math.exp %scalar_input_exp : f32
          memref.store %scalar_result_exp, %alloc_2[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
        }
      }
    }
    memref.dealloc %alloc_1 : memref<1x40x151936xf32>
    
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x40xf32>
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        memref.store %cst, %alloc_3[%arg1, %arg2] : memref<1x40xf32>
      }
    }
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        scf.for %arg3 = %c0 to %c151936 step %c1 {
          %3 = memref.load %alloc_2[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
          %4 = memref.load %alloc_3[%arg1, %arg2] : memref<1x40xf32>
          %5 = arith.addf %3, %4 : f32 
          memref.store %5, %alloc_3[%arg1, %arg2] : memref<1x40xf32>
        }
      }
    }
    memref.dealloc %alloc_2 : memref<1x40x151936xf32>
    
    %reinterpret_cast_4 = memref.reinterpret_cast %alloc_3 to offset: [0], sizes: [1, 40, 1], strides: [40, 1, 1] : memref<1x40xf32> to memref<1x40x1xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x40x1xf32>
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %3 = memref.load %reinterpret_cast_4[%c0, %arg2, %c0] : memref<1x40x1xf32>
          %4 = math.log %3 : f32 
          memref.store %4, %alloc_5[%arg1, %arg2, %arg3] : memref<1x40x1xf32>
        }
      }
    }
    memref.dealloc %alloc_3 : memref<1x40xf32>
    
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x40x1xf32>
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %3 = memref.load %reinterpret_cast[%c0, %arg2, %c0] : memref<1x40x1xf32>
          %4 = memref.load %alloc_5[%c0, %arg2, %c0] : memref<1x40x1xf32>
          %5 = arith.addf %3, %4 : f32
          memref.store %5, %alloc_6[%arg1, %arg2, %arg3] : memref<1x40x1xf32>
        }
      }
    }
    memref.dealloc %alloc_5 : memref<1x40x1xf32>
    memref.dealloc %alloc : memref<1x40xf32>
    
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x40x151936xf32>

    %remainder_step7 = arith.remsi %c151936, %c16 : index
    %vectorized_end_step7 = arith.subi %c151936, %remainder_step7 : index
    
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        %logsumexp_scalar = memref.load %alloc_6[%c0, %arg2, %c0] : memref<1x40x1xf32>
        %logsumexp_vector = vector.splat %logsumexp_scalar : vector<16xf32>
        scf.for %arg3 = %c0 to %vectorized_end_step7 step %c16 {
          %input_vector_step7 = vector.load %arg0[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>, vector<16xf32>
          %result_vector_step7 = arith.subf %input_vector_step7, %logsumexp_vector : vector<16xf32>
          vector.store %result_vector_step7, %alloc_7[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>, vector<16xf32>
        }
        scf.for %arg3 = %vectorized_end_step7 to %c151936 step %c1 {
          %scalar_input_step7 = memref.load %arg0[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
          %scalar_result_step7 = arith.subf %scalar_input_step7, %logsumexp_scalar : f32
          memref.store %scalar_result_step7, %alloc_7[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
        }
      }
    }
    memref.dealloc %alloc_6 : memref<1x40x1xf32>
    
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x40x151936xf32>
    %remainder_step8 = arith.remsi %c151936, %c16 : index
    %vectorized_end_step8 = arith.subi %c151936, %remainder_step8 : index
    
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c40 step %c1 {
        scf.for %arg3 = %c0 to %vectorized_end_step8 step %c16 {
          %input_vector_step8 = vector.load %alloc_7[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>, vector<16xf32>
          %softmax_vector = math.exp %input_vector_step8 : vector<16xf32>
          vector.store %softmax_vector, %alloc_8[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>, vector<16xf32>
        }
        scf.for %arg3 = %vectorized_end_step8 to %c151936 step %c1 {
          %scalar_input_step8 = memref.load %alloc_7[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
          %scalar_softmax = math.exp %scalar_input_step8 : f32
          memref.store %scalar_softmax, %alloc_8[%arg1, %arg2, %arg3] : memref<1x40x151936xf32>
        }
      }
    }
    memref.dealloc %alloc_7 : memref<1x40x151936xf32>
    %1 = call @rtclock() : () -> f64
    %2 = arith.subf %1, %0 : f64
    %cast = memref.cast %alloc_8 : memref<1x40x151936xf32> to memref<*xf32>

    // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 151936] strides = [6077440, 151936, 1] data =

    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %alloc_8 : memref<1x40x151936xf32>
    vector.print %2 : f64
    return
  }
  func.func @main() {
    %0 = memref.get_global @__constant_1x40x151936xf32 : memref<1x40x151936xf32>
    call @softmax_kernel(%0) : (memref<1x40x151936xf32>) -> ()
    return
  }
}

