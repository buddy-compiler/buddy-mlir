// Batch matmul from DeepSeek attention: Attn_scores × V
// Shape: [12, 1024, 1024] × [12, 1024, 128] → [12, 1024, 128]
// 28 calls per prefill (one per layer), currently 1 thread/block → 11.8s total
func.func @batch_matmul_attn(%A: memref<12x1024x1024xf32>,
                              %B: memref<12x1024x128xf32>,
                              %C: memref<12x1024x128xf32>) {
  linalg.batch_matmul ins(%A, %B : memref<12x1024x1024xf32>, memref<12x1024x128xf32>)
                      outs(%C : memref<12x1024x128xf32>)
  return
}
