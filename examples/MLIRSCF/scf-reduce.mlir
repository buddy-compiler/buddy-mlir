// RUN: buddy-opt %s \
// RUN:     -async-parallel-for \
// RUN:     -async-to-async-runtime \
// RUN:     -async-runtime-ref-counting \
// RUN:     -async-runtime-ref-counting-opt \
// RUN:     -arith-expand \
// RUN:     -convert-async-to-llvm \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-scf-to-cf \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_async_runtime%shlibext \
// RUN: | FileCheck %s

memref.global "private" @gv : memref<16xf32> = dense<[1., 2., 3., 4.,5., 6., 7., 8.,
                                                      9., 10., 11., 12.,13., 14., 15., 16.]>

func.func @main() {
  %c0 = arith.constant 0.0 : f32
  %index_0 = arith.constant 0 : index
  %index_1 = arith.constant 1 : index
  %index_16 = arith.constant 16 : index
  %mem = memref.get_global @gv : memref<16xf32>
  %res = scf.parallel (%i0) = (%index_0) to (%index_16)
                                       step (%index_1) init (%c0) -> f32 {
    %v = memref.load %mem[%i0] : memref<16xf32>
    scf.reduce(%v)  : f32 {
      ^bb0(%lhs: f32, %rhs: f32):
        %0 = arith.addf %lhs, %rhs : f32
        scf.reduce.return %0 : f32
    }
  }
  // CHECK: 136
  vector.print %res : f32
  return
}
