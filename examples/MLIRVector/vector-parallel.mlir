// RUN: mlir-opt %s \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-openmp \
// RUN:     -cse -memref-expand -arith-expand \
// RUN:     -convert-vector-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf -convert-cf-to-llvm -convert-openmp-to-llvm \
// RUN:     -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN:     -shared-libs=%openmp_runtime_dir/libomp%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @init_data(%A: memref<512x512xf32>, %B: memref<512x512xf32>) {
    %one = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%i, %j) = (0, 0) to (512, 512) {
      affine.store %one, %A[%i, %j] : memref<512x512xf32>
    }
    affine.parallel (%i, %j) = (0, 0) to (512, 512) {
      %on_diag = arith.cmpi eq, %i, %j : index
      %diag_val = arith.addi %j, %c1 : index
      %diag_i32 = arith.index_cast %diag_val : index to i32
      %diag_f = arith.sitofp %diag_i32 : i32 to f32
      %b_ij = arith.select %on_diag, %diag_f, %zero : f32
      affine.store %b_ij, %B[%i, %j] : memref<512x512xf32>
    }
    return
  }

  // 512x512 workload, VF=16 (32 vector blocks per row).
  func.func @gemm_tile_parallel(%A: memref<512x512xf32>, %B: memref<512x512xf32>,
                                %C: memref<512x512xf32>) {
    %zero = arith.constant 0.0 : f32
    affine.parallel (%i, %j) = (0, 0) to (512, 512) {
      affine.store %zero, %C[%i, %j] : memref<512x512xf32>
    }
    affine.parallel (%i, %j) = (0, 0) to (512, 512) step (1, 16) {
      %acc = vector.load %C[%i, %j] : memref<512x512xf32>, vector<16xf32>
      %result = affine.for %k = 0 to 512 iter_args(%acc_iter = %acc) -> (vector<16xf32>) {
        %a = affine.load %A[%i, %k] : memref<512x512xf32>
        %a_vec = vector.broadcast %a : f32 to vector<16xf32>
        %b_vec = vector.load %B[%k, %j] : memref<512x512xf32>, vector<16xf32>
        %fma = vector.fma %a_vec, %b_vec, %acc_iter : vector<16xf32>
        affine.yield %fma : vector<16xf32>
      }
      vector.store %result, %C[%i, %j] : memref<512x512xf32>, vector<16xf32>
    }
    return
  }

  func.func @main() {
    %A = memref.alloc() : memref<512x512xf32>
    %B = memref.alloc() : memref<512x512xf32>
    %C = memref.alloc() : memref<512x512xf32>
    func.call @init_data(%A, %B) : (memref<512x512xf32>, memref<512x512xf32>) -> ()
    func.call @gemm_tile_parallel(%A, %B, %C)
      : (memref<512x512xf32>, memref<512x512xf32>, memref<512x512xf32>) -> ()
    %out = memref.cast %C : memref<512x512xf32> to memref<*xf32>
    // CHECK: sizes = [512, 512]
    // CHECK-NEXT: {{.*}}1, {{.*}}16, {{.*}}17, {{.*}}32, {{.*}}512]
    func.call @printMemrefF32(%out) : (memref<*xf32>) -> ()
    return
  }
}
