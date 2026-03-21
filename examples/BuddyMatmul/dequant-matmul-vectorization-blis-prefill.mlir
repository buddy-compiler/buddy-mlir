// RUN: buddy-opt %s \
// RUN:     -matmul-vectorization-blis \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -expand-strided-metadata \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
//
// Prefill-style W8A32 dequant + matmul pattern (M > 1).
// This is a minimal example that exercises the dequant fusion inside
// matmul-vectorization-blis (B-packing reads i8 weights and applies scale).
//
// Pattern:
//   cast_generic (i8->f32) -> cast_buf
//   mul_generic  (* scale) -> dequant_buf
//   linalg.matmul(A_f32[M,K], dequant_buf[K,N]) -> C_f32[M,N]

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @prefill_w8a32(
      %A: memref<4x4xf32>,
      %Wi8: memref<4x4xi8>,
      %scale: memref<1x4xf32>,
      %Cinit: memref<4x4xf32>) -> memref<4x4xf32> {
    // Intermediate buffers.
    %cast_buf = memref.alloc() : memref<4x4xf32>
    %dequant_buf = memref.alloc() : memref<4x4xf32>

    // cast_generic: i8 -> f32
    linalg.generic
      {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%Wi8 : memref<4x4xi8>)
      outs(%cast_buf : memref<4x4xf32>) {
      ^bb0(%in: i8, %out: f32):
        %f = arith.sitofp %in : i8 to f32
        linalg.yield %f : f32
    }

    // mul_generic: cast_buf * scale (broadcast along N).
    linalg.generic
      {indexing_maps = [
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(0,j)>,
          affine_map<(i,j)->(i,j)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%cast_buf, %scale : memref<4x4xf32>, memref<1x4xf32>)
      outs(%dequant_buf : memref<4x4xf32>) {
      ^bb0(%w: f32, %s: f32, %out: f32):
        %p = arith.mulf %w, %s : f32
        linalg.yield %p : f32
    }

    // Matmul (prefill-style: M>1).
    linalg.matmul
      ins(%A, %dequant_buf : memref<4x4xf32>, memref<4x4xf32>)
      outs(%Cinit : memref<4x4xf32>)

    return %Cinit : memref<4x4xf32>
  }

  // Simple driver to JIT-run the BLIS prefill-style pattern.
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    %A = memref.alloc() : memref<4x4xf32>
    %Wi8 = memref.alloc() : memref<4x4xi8>
    %scale = memref.alloc() : memref<1x4xf32>
    %C = memref.alloc() : memref<4x4xf32>

    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    %two_i8 = arith.constant 2 : i8

    // A = 1.0, Wi8 = 2, scale = 1.0, C = 0.0
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        memref.store %one, %A[%i, %j] : memref<4x4xf32>
        memref.store %two_i8, %Wi8[%i, %j] : memref<4x4xi8>
      }
    }
    scf.for %j = %c0 to %c4 step %c1 {
      memref.store %one, %scale[%c0, %j] : memref<1x4xf32>
    }
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        memref.store %zero, %C[%i, %j] : memref<4x4xf32>
      }
    }

    %Cout = func.call @prefill_w8a32(%A, %Wi8, %scale, %C)
              : (memref<4x4xf32>, memref<4x4xi8>, memref<1x4xf32>, memref<4x4xf32>) -> memref<4x4xf32>

    %Ccast = memref.cast %Cout : memref<4x4xf32> to memref<*xf32>
    func.call @printMemrefF32(%Ccast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
    // CHECK-NEXT: [
    // CHECK: [8, 8, 8, 8]

    return
  }
}
