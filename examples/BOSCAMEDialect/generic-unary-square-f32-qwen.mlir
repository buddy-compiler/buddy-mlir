// RUN: buddy-opt %s -lower-linalg-to-boscame
//
// This file tests the Qwen3 RMSNorm-style linalg.generic pattern:
//   y = fpowi(x, 2)
// over a rank-3 tensor-shaped memref. The BOSCAME lowering treats the last two
// dimensions as the matrix tile and emits outer loops for leading dimensions.
//

#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func private @print_square(f32, f32, f32, f32)

  func.func @generic_square_f32_1x4x4(%A: memref<1x4x4xf32>,
                                      %C: memref<1x4x4xf32>) {
    %c2_i32 = arith.constant 2 : i32
    linalg.generic {
      indexing_maps = [#map3, #map3],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%A : memref<1x4x4xf32>)
      outs(%C : memref<1x4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sq = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %sq : f32
    }
    return
  }

  func.func @main() -> i32 {
    %A = memref.alloc() : memref<1x4x4xf32>
    %C = memref.alloc() : memref<1x4x4xf32>

    %c0 = arith.constant 0.000000e+00 : f32
    %c2 = arith.constant 2.000000e+00 : f32
    %c3 = arith.constant 3.000000e+00 : f32
    %c4 = arith.constant 4.000000e+00 : f32
    %c5 = arith.constant 5.000000e+00 : f32

    linalg.fill ins(%c2 : f32) outs(%A : memref<1x4x4xf32>)
    linalg.fill ins(%c0 : f32) outs(%C : memref<1x4x4xf32>)

    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

    memref.store %c3, %A[%i0, %i0, %i1] : memref<1x4x4xf32>
    memref.store %c4, %A[%i0, %i0, %i2] : memref<1x4x4xf32>
    memref.store %c5, %A[%i0, %i0, %i3] : memref<1x4x4xf32>

    call @generic_square_f32_1x4x4(%A, %C)
      : (memref<1x4x4xf32>, memref<1x4x4xf32>) -> ()

    %v0 = memref.load %C[%i0, %i0, %i0] : memref<1x4x4xf32>
    %v1 = memref.load %C[%i0, %i0, %i1] : memref<1x4x4xf32>
    %v2 = memref.load %C[%i0, %i0, %i2] : memref<1x4x4xf32>
    %v3 = memref.load %C[%i0, %i0, %i3] : memref<1x4x4xf32>
    call @print_square(%v0, %v1, %v2, %v3) : (f32, f32, f32, f32) -> ()

    memref.dealloc %A : memref<1x4x4xf32>
    memref.dealloc %C : memref<1x4x4xf32>

    %ret = arith.constant 0 : i32
    return %ret : i32
  }
}
