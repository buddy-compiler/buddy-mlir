// RUN: buddy-opt %s \
// RUN:     -pass-pipeline="builtin.module(func.func(scf-parallel-loop-fusion))" \
// RUN: | buddy-opt \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-openmp\
// RUN:   -one-shot-bufferize \
// RUN:   -buffer-deallocation \
// RUN:   -expand-strided-metadata \
// RUN:   -convert-vector-to-llvm \
// RUN:   -memref-expand \
// RUN:   -arith-expand \
// RUN:   -convert-arith-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-openmp-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-math-to-llvm \
// RUN:   -convert-math-to-libm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libomp%shlibext \
// RUN: | FileCheck %s

module{
  memref.global "private" constant @input_1x40x151936xf32 : memref<1x40x151936xf32> = dense<3.000000e+00> {alignment = 64 : i64}
  func.func private @rtclock() -> f64
  func.func private @printMemrefF32(memref<*xf32>)

  llvm.func @llvm.prefetch(!llvm.ptr, i32, i32, i32) -> ()

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

          "scf.reduce"() :()->()
        }

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 40, 151936] strides = [6077440, 151936, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [0.952574{{(, 0.952574)*}}],

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    %cast = memref.cast %sigmoid : memref<1x40x151936xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
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

