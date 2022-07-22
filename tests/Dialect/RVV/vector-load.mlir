//
// x86
//
// RUN: mlir-opt %s --convert-vector-to-scf --lower-affine --convert-scf-to-cf --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts  \
// RUN: | mlir-cpu-runner -O0 -e main -entry-point-result=i32 \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
//
// RVV
//
// RUN: mlir-opt %s --convert-vector-to-scf --lower-affine --convert-scf-to-cf --convert-vector-to-llvm \
// RUN:             --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-translate --mlir-to-llvmir \
// RUN: | qemu-riscv64 -L %riscv_gnu_toolchain_sysroot -cpu rv64,x-v=true,vlen=128 \
// RUN: %cross_lli_bin --march=riscv64 -mattr=+m,+d,+v -jit-linker=jitlink -relocation-model=pic \
// RUN: --dlopen=%cross_mlir_runner_utils/libmlir_c_runner_utils.so | FileCheck %s

memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // Load 1-D vector from memref.
  %vec_1d = vector.load %mem[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
  vector.print %vec_1d : vector<4xf32>
// CHECK: ( 0, 1, 2, 3 )
  // Load 2-D vector from memref.
  %f0 = arith.constant 0.0 : f32
  %vec_2d = vector.transfer_read %mem[%c2, %c0], %f0
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} :
    memref<4x4xf32>, vector<2x4xf32>
  vector.print %vec_2d : vector<2x4xf32>
// CHECK: ( ( 20, 21, 22, 23 ), ( 30, 31, 32, 33 ) )

  %ret = arith.constant 0 : i32
  return %ret : i32
}
