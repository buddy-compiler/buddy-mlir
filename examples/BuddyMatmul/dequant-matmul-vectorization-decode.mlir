// RUN: buddy-opt %s \
// RUN:     -dequant-matmul-vectorization-decode=vector-size=8 \
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
// Simple decode-style W8A32 dequant + matmul pattern (m = 1).
// This is a minimal example for the dequant-matmul-vectorization-decode pass.
//
// Pattern:
//   cast_generic (sitofp i8 -> f32) -> cast_buf
//   mul_generic  (cast_buf * scale) -> dequant_buf
//   linalg.matmul(A_f32, dequant_buf) -> C_f32

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @decode_w8a32(
      %A: memref<1x8xf32>,
      %Wi8: memref<8x8xi8>,
      %scale: memref<1x8xf32>) -> memref<1x8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index

    // Allocate intermediate buffers.
    %cast_buf = memref.alloc() : memref<8x8xf32>
    %dequant_buf = memref.alloc() : memref<8x8xf32>
    %C = memref.alloc() : memref<1x8xf32>

    // cast_generic: i8 -> f32
    linalg.generic
      {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%Wi8 : memref<8x8xi8>)
      outs(%cast_buf : memref<8x8xf32>) {
      ^bb0(%in: i8, %out: f32):
        %f = arith.sitofp %in : i8 to f32
        linalg.yield %f : f32
    }

    // mul_generic: cast_buf * scale (broadcast along N)
    linalg.generic
      {indexing_maps = [
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(0,j)>,
          affine_map<(i,j)->(i,j)>
       ],
       iterator_types = ["parallel", "parallel"]}
      ins(%cast_buf, %scale : memref<8x8xf32>, memref<1x8xf32>)
      outs(%dequant_buf : memref<8x8xf32>) {
      ^bb0(%w: f32, %s: f32, %out: f32):
        %p = arith.mulf %w, %s : f32
        linalg.yield %p : f32
    }

    // Initialize C with zeros.
    %zero = arith.constant 0.0 : f32
    scf.for %j = %c0 to %c1 step %c1 {
      scf.for %k = %c0 to %c8 step %c1 {
        memref.store %zero, %C[%j, %k] : memref<1x8xf32>
      }
    }

    // Matmul (decode-style: m=1).
    linalg.matmul
      ins(%A, %dequant_buf : memref<1x8xf32>, memref<8x8xf32>)
      outs(%C : memref<1x8xf32>)

    return %C : memref<1x8xf32>
  }

  // Simple driver to JIT-run the pattern.
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index

    %A = memref.alloc() : memref<1x8xf32>
    %Wi8 = memref.alloc() : memref<8x8xi8>
    %scale = memref.alloc() : memref<1x8xf32>

    // Initialize A with 1.0, Wi8 with small ints, scale with 0.5.
    %one = arith.constant 1.0 : f32
    %half = arith.constant 5.000000e-01 : f32
    %two_i8 = arith.constant 2 : i8

    scf.for %i = %c0 to %c1 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        memref.store %one, %A[%i, %j] : memref<1x8xf32>
        memref.store %two_i8, %Wi8[%j, %i] : memref<8x8xi8>
        memref.store %half, %scale[%i, %j] : memref<1x8xf32>
      }
    }

    %C = func.call @decode_w8a32(%A, %Wi8, %scale)
           : (memref<1x8xf32>, memref<8x8xi8>, memref<1x8xf32>) -> memref<1x8xf32>

    %Ccast = memref.cast %C : memref<1x8xf32> to memref<*xf32>
    func.call @printMemrefF32(%Ccast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [1, 8] strides = [8, 1] data =
    // CHECK-NEXT: [
    // CHECK-NEXT:   [8, 8, 0, 0, 0, 0, 0, 0]
    // CHECK-NEXT: ]

    return
  }
}
