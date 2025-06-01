// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-llvm \
// RUN:     -memref-expand \
// RUN:     -arith-expand \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext

module {
  memref.global "private" constant @__constant_3072x1536xf32 : memref<3072x1536xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_40x3072xf32_0 : memref<40x3072xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_40x1536xf32 : memref<40x1536xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  
  func.func @kernel(%arg0: memref<40x3072xf32>, %arg1: memref<3072x1536xf32>) {
    %c3072 = arith.constant 3072 : index
    %c1536 = arith.constant 1536 : index
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index  
    %c0_f32 = arith.constant 0.000000e+00 : f32
    
    %0 = call @rtclock() : () -> f64

    %alloc = memref.alloc() {alignment = 64 : i64} : memref<40x1536xf32>
    
    %zero_vec = vector.broadcast %c0_f32 : f32 to vector<16xf32>
    scf.for %i = %c0 to %c40 step %c1 {
      scf.for %j = %c0 to %c1536 step %c16 {
        vector.store %zero_vec, %alloc[%i, %j] : memref<40x1536xf32>, vector<16xf32>
      }
    }
    scf.for %arg2 = %c0 to %c40 step %c1 {
      scf.for %arg3 = %c0 to %c1536 step %c16 {
        %acc = vector.load %alloc[%arg2, %arg3] : memref<40x1536xf32>, vector<16xf32>
        
        %final_acc = scf.for %arg4 = %c0 to %c3072 step %c1 iter_args(%acc_iter = %acc) -> (vector<16xf32>) {
          %a_scalar = memref.load %arg0[%arg2, %arg4] : memref<40x3072xf32>
          %a_vec = vector.broadcast %a_scalar : f32 to vector<16xf32>
          
          %b_vec = vector.load %arg1[%arg4, %arg3] : memref<3072x1536xf32>, vector<16xf32>
          
          %new_acc = vector.fma %a_vec, %b_vec, %acc_iter : vector<16xf32>
          scf.yield %new_acc : vector<16xf32>
        }
        
        vector.store %final_acc, %alloc[%arg2, %arg3] : memref<40x1536xf32>, vector<16xf32>
      }
    }
    
    %2 = call @rtclock() : () -> f64
    %3 = arith.subf %2, %0 : f64
    %cast = memref.cast %alloc : memref<40x1536xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<40x1536xf32>
    vector.print %3 : f64
    return
  }
  
  func.func @main() {
    %0 = memref.get_global @__constant_40x3072xf32_0 : memref<40x3072xf32>
    %1 = memref.get_global @__constant_3072x1536xf32 : memref<3072x1536xf32>
    call @kernel(%0, %1) : (memref<40x3072xf32>, memref<3072x1536xf32>) -> ()
    return
  }
}
