// next-sigmoid-manual-run:
// 	@${BUDDY_OPT} ./next-sigmoid-manual.mlir \
// 	     -pass-pipeline="builtin.module(func.func(scf-parallel-loop-fusion))" | \
// 	${BUDDY_OPT} \
//   	-lower-affine \
//   	-convert-scf-to-openmp\
//   	-buffer-deallocation \
//   	-finalizing-bufferize \
//   	-expand-strided-metadata \
//   	-convert-vector-to-llvm \
//   	-memref-expand \
//   	-arith-expand \
//   	-convert-arith-to-llvm \
//   	-finalize-memref-to-llvm \
//   	-convert-scf-to-cf \
//   	-convert-openmp-to-llvm \
//   	-convert-arith-to-llvm \
//   	-convert-math-to-llvm \
//   	-convert-math-to-libm \
//   	-convert-func-to-llvm \
//   	-reconcile-unrealized-casts| \
// 	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void\
// 		-shared-libs=${OPENMP_PATH}/libomp.so\
// 	    -shared-libs=${MLIR_RUNNER_UTILS} \
// 	    -shared-libs=${MLIR_C_RUNNER_UTILS}\


module attributes {
  llvm.target_triple = "x86_64-unknown-linux-gnu",
  llvm.target_cpu = "broadwell", 
  llvm.target_features = "+avx2,+fma,+bmi,+bmi2,+popcnt" 
}{
  memref.global "private" constant @input_1x40x151936xf32 : memref<1x40x151936xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)
  // LLVM预取函数声明
  llvm.func @llvm.prefetch(!llvm.ptr<i8>, i32, i32, i32) -> ()

  func.func @kenerl(%arg0: memref<1x40x151936xf32>) {
   
    %sigmoid = memref.alloc() {alignment = 64 : i64} : memref<1x40x151936xf32>
    %1 = arith.constant 1.0 :f32
    %zero = arith.constant 0.0 :f32
    %ss_0 = arith.constant 0 :index
    %ss_1 = arith.constant 1 :index
    %ss_4 = arith.constant 4:index
    %ss_8 = arith.constant 8:index
    %ss_10 = arith.constant 10:index
    %ss_16 = arith.constant 16 :index
    %ss_32 = arith.constant 32:index
    %ss_40 = arith.constant 40:index
    %ss_151936 = arith.constant 151936:index

    // 预取参数（read=0, locality=0, cache_type=0）
    %cst_i32_0 = arith.constant 0 : i32

    %t_start = call @rtclock() : () -> f64
    
        scf.parallel (%j,%k) = (%ss_0,%ss_0) to (%ss_40,%ss_151936) step (%ss_1,%ss_8){
          %next_k = arith.addi %k,%ss_8 : index
          %in_bound = arith.cmpi slt, %next_k,%ss_151936 :index
          scf.if %in_bound{
            %zero_i64 = arith.constant 0 :i64
            %j_i64 = arith.index_cast %j : index to i64
            %next_k_i64 = arith.index_cast %next_k :index to i64
            memref.prefetch %arg0[%ss_0,%j,%next_k] ,read,locality<2>,data :memref<1x40x151936xf32>
          }

          %x = vector.load %arg0[%ss_0,%j,%k] :memref<1x40x151936xf32> ,vector<8xf32>
          %neg_x = arith.negf %x : vector<8xf32>
          %exp = math.exp %neg_x :vector<8xf32>
          %one = vector.broadcast %1 : f32 to vector<8xf32>
          %denom = arith.addf %exp, %one : vector<8xf32>
          %result = arith.divf %one, %denom : vector<8xf32>
          
          vector.store %result,%sigmoid[%ss_0,%j,%k] : memref<1x40x151936xf32>,vector<8xf32>

          scf.yield
        }
        

    
    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    %cast = memref.cast %sigmoid : memref<1x40x151936xf32> to memref<*xf32>
    //call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    vector.print %time : f64
    
    memref.dealloc %sigmoid : memref<1x40x151936xf32>
    return
  }

  func.func @main() {
    %0 = memref.get_global @input_1x40x151936xf32 : memref<1x40x151936xf32>
    call @kenerl(%0) : (memref<1x40x151936xf32>) -> ()
    return
  }
}
