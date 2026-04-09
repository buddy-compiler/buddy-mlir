// RUN: buddy-opt %s \
// RUN:     -int4-dequant-matmul-vectorization-decode=vector-size=8 \
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
// W4A16-style decode pattern with int4 unpack + dequant + matmul (m = 1).
// This is a minimal example for the int4-dequant-matmul-vectorization-decode pass.
//
// Rough structure:
//   packed_i8[K,N/2] --bitwise/shift--> low/high --> reshape/concat/reshape
//      -> unpacked_i8[K,N]
//      -> cast_generic (i8->f16)
//      -> mul_generic  (* scale)
//      -> linalg.matmul(A_f16, dequant_buf) -> C_f16

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @decode_w4a16(
      %A: memref<1x4xf16>,
      %packed: memref<4x2xi8>,
      %scale: memref<1x4xf16>) -> memref<1x4xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Constants for unpack.
    %mask = memref.alloc() : memref<4x2xi8>
    %shift4 = memref.alloc() : memref<4x2xi8>
    %c15_i8 = arith.constant 15 : i8
    %c4_i8 = arith.constant 4 : i8
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        memref.store %c15_i8, %mask[%i, %j] : memref<4x2xi8>
        memref.store %c4_i8, %shift4[%i, %j] : memref<4x2xi8>
      }
    }

    // Low and high nibbles.
    %low_masked = memref.alloc() : memref<4x2xi8>
    %low_shifted = memref.alloc() : memref<4x2xi8>
    %low = memref.alloc() : memref<4x2xi8>
    %high = memref.alloc() : memref<4x2xi8>

    linalg.generic
      {indexing_maps = [
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>],
       iterator_types = ["parallel","parallel"]}
      ins(%packed, %mask : memref<4x2xi8>, memref<4x2xi8>)
      outs(%low_masked : memref<4x2xi8>) {
      ^bb0(%p: i8, %m: i8, %out: i8):
        %v = arith.andi %p, %m : i8
        linalg.yield %v : i8
    }

    linalg.generic
      {indexing_maps = [
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>],
       iterator_types = ["parallel","parallel"]}
      ins(%low_masked, %shift4 : memref<4x2xi8>, memref<4x2xi8>)
      outs(%low_shifted : memref<4x2xi8>) {
      ^bb0(%p: i8, %s: i8, %out: i8):
        %v = arith.shli %p, %s : i8
        linalg.yield %v : i8
    }

    linalg.generic
      {indexing_maps = [
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>],
       iterator_types = ["parallel","parallel"]}
      ins(%low_shifted, %shift4 : memref<4x2xi8>, memref<4x2xi8>)
      outs(%low : memref<4x2xi8>) {
      ^bb0(%p: i8, %s: i8, %out: i8):
        %v = arith.shrsi %p, %s : i8
        linalg.yield %v : i8
    }

    linalg.generic
      {indexing_maps = [
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>],
       iterator_types = ["parallel","parallel"]}
      ins(%packed, %shift4 : memref<4x2xi8>, memref<4x2xi8>)
      outs(%high : memref<4x2xi8>) {
      ^bb0(%p: i8, %s: i8, %out: i8):
        %v = arith.shrsi %p, %s : i8
        linalg.yield %v : i8
    }

    // Interleave low/high into [4x4xi8]. To keep it simple, materialize into
    // a single buffer via scalar loops.
    %unpacked = memref.alloc() : memref<4x4xi8>
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %b0 = memref.load %low[%i, %j] : memref<4x2xi8>
        %b1 = memref.load %high[%i, %j] : memref<4x2xi8>
        %j2 = arith.muli %j, %c2 : index
        %j2p1 = arith.addi %j2, %c1 : index
        memref.store %b0, %unpacked[%i, %j2] : memref<4x4xi8>
        memref.store %b1, %unpacked[%i, %j2p1] : memref<4x4xi8>
      }
    }

    // Dequant: cast + mul.
    %cast_buf = memref.alloc() : memref<4x4xf16>
    %dequant_buf = memref.alloc() : memref<4x4xf16>

    linalg.generic
      {indexing_maps = [
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(i,j)>],
       iterator_types = ["parallel","parallel"]}
      ins(%unpacked : memref<4x4xi8>)
      outs(%cast_buf : memref<4x4xf16>) {
      ^bb0(%in: i8, %out: f16):
        %f = arith.sitofp %in : i8 to f16
        linalg.yield %f : f16
    }

    linalg.generic
      {indexing_maps = [
          affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(0,j)>,
          affine_map<(i,j)->(i,j)>],
       iterator_types = ["parallel","parallel"]}
      ins(%cast_buf, %scale : memref<4x4xf16>, memref<1x4xf16>)
      outs(%dequant_buf : memref<4x4xf16>) {
      ^bb0(%w: f16, %s: f16, %out: f16):
        %p = arith.mulf %w, %s : f16
        linalg.yield %p : f16
    }

    // Matmul (decode-style: m=1).
    %C = memref.alloc() : memref<1x4xf16>
    linalg.matmul
      ins(%A, %dequant_buf : memref<1x4xf16>, memref<4x4xf16>)
      outs(%C : memref<1x4xf16>)

    return %C : memref<1x4xf16>
  }

  // Simple driver to JIT-run the int4 decode pattern.
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    %A = memref.alloc() : memref<1x4xf16>
    %packed = memref.alloc() : memref<4x2xi8>
    %scale = memref.alloc() : memref<1x4xf16>

    %one = arith.constant 1.0 : f16
    %two_i8 = arith.constant 2 : i8

    // A = 1.0, packed bytes = 2, scale = 1.0
    scf.for %j = %c0 to %c4 step %c1 {
      memref.store %one, %A[%c0, %j] : memref<1x4xf16>
      memref.store %one, %scale[%c0, %j] : memref<1x4xf16>
    }
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        memref.store %two_i8, %packed[%i, %j] : memref<4x2xi8>
      }
    }

    %C = func.call @decode_w4a16(%A, %packed, %scale)
           : (memref<1x4xf16>, memref<4x2xi8>, memref<1x4xf16>) -> memref<1x4xf16>

    // Cast C (f16) into a temporary f32 buffer for printing.
    %C_f32 = memref.alloc() : memref<1x4xf32>
    %c0_i = arith.constant 0 : index
    %c1_i = arith.constant 1 : index
    %c4_i = arith.constant 4 : index
    scf.for %j = %c0_i to %c4_i step %c1_i {
      %v = memref.load %C[%c0_i, %j] : memref<1x4xf16>
      %v_f32 = arith.extf %v : f16 to f32
      memref.store %v_f32, %C_f32[%c0_i, %j] : memref<1x4xf32>
    }
    %Ccast = memref.cast %C_f32 : memref<1x4xf32> to memref<*xf32>
    func.call @printMemrefF32(%Ccast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [1, 4] strides = [4, 1] data =
    // CHECK-NEXT: [
    return
  }
}
