// Decode bottleneck: broadcast add for attention KV cache
// Shape: [1,2,1,1024,128] + [1,2,6,1024,128] → [1,2,6,1024,128]
// In e2e: grid=(2,6,1024) block=(1,1,1) with scf.for over 128 → 42.8% of decode time
//
// This is a linalg.generic that broadcasts dim 2 (group) of A.
// After convert-linalg-to-parallel-loops, all dims are scf.parallel except
// the innermost (128) which becomes scf.for because gpu-map-parallel-loops
// only maps 3 dims to blocks and ignores the rest.

func.func @broadcast_add(%A: memref<2x1x1024x128xf32>,
                          %B: memref<2x6x1024x128xf32>,
                          %C: memref<2x6x1024x128xf32>) {
  // Simplified: removed batch=1 dim for clarity.
  // This is the core linalg.generic from the e2e graph.
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>,  // A: broadcast d1
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,  // B
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>   // C
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%A, %B : memref<2x1x1024x128xf32>, memref<2x6x1024x128xf32>)
    outs(%C : memref<2x6x1024x128xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum : f32
  }
  return
}
