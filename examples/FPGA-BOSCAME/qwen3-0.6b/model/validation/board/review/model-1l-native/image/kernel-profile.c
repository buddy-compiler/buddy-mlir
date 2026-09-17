/* Generated profiling glue; calls the linked Triton kernel. */
#include "support.h"
#include "nr_runtime.h"
static uint64_t cycles[46], counts[46];
void qwen_profile_reset(void) {
  for (unsigned i=0; i<46; ++i) cycles[i]=counts[i]=0;
}
void qwen_profile_report(unsigned position) {
  if (counts[0]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_attention_pv_position_16x16x128x512 calls="); nr_hex64(counts[0]);
    nr_puts(" cycles="); nr_hex64(cycles[0]);
    nr_puts("\r\n");
  }
  if (counts[1]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_attention_pv_position_16x1x128x512 calls="); nr_hex64(counts[1]);
    nr_puts(" cycles="); nr_hex64(cycles[1]);
    nr_puts("\r\n");
  }
  if (counts[2]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_attention_qk_position_native_16x16x512x128 calls="); nr_hex64(counts[2]);
    nr_puts(" cycles="); nr_hex64(cycles[2]);
    nr_puts("\r\n");
  }
  if (counts[3]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_attention_qk_position_native_16x1x512x128 calls="); nr_hex64(counts[3]);
    nr_puts(" cycles="); nr_hex64(cycles[3]);
    nr_puts("\r\n");
  }
  if (counts[4]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_attention_scale_mask_position_16x16x512 calls="); nr_hex64(counts[4]);
    nr_puts(" cycles="); nr_hex64(cycles[4]);
    nr_puts("\r\n");
  }
  if (counts[5]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_attention_scale_mask_position_16x1x512 calls="); nr_hex64(counts[5]);
    nr_puts(" cycles="); nr_hex64(cycles[5]);
    nr_puts("\r\n");
  }
  if (counts[6]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_dequantize_16x1024 calls="); nr_hex64(counts[6]);
    nr_puts(" cycles="); nr_hex64(cycles[6]);
    nr_puts("\r\n");
  }
  if (counts[7]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_dequantize_16x2048 calls="); nr_hex64(counts[7]);
    nr_puts(" cycles="); nr_hex64(cycles[7]);
    nr_puts("\r\n");
  }
  if (counts[8]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_dequantize_16x3072 calls="); nr_hex64(counts[8]);
    nr_puts(" cycles="); nr_hex64(cycles[8]);
    nr_puts("\r\n");
  }
  if (counts[9]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_dequantize_1x1024 calls="); nr_hex64(counts[9]);
    nr_puts(" cycles="); nr_hex64(cycles[9]);
    nr_puts("\r\n");
  }
  if (counts[10]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_dequantize_1x151936 calls="); nr_hex64(counts[10]);
    nr_puts(" cycles="); nr_hex64(cycles[10]);
    nr_puts("\r\n");
  }
  if (counts[11]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_dequantize_1x2048 calls="); nr_hex64(counts[11]);
    nr_puts(" cycles="); nr_hex64(cycles[11]);
    nr_puts("\r\n");
  }
  if (counts[12]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_dequantize_1x3072 calls="); nr_hex64(counts[12]);
    nr_puts(" cycles="); nr_hex64(cycles[12]);
    nr_puts("\r\n");
  }
  if (counts[13]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_embedding_w8a8_16x1024 calls="); nr_hex64(counts[13]);
    nr_puts(" cycles="); nr_hex64(cycles[13]);
    nr_puts("\r\n");
  }
  if (counts[14]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_embedding_w8a8_1x1024 calls="); nr_hex64(counts[14]);
    nr_puts(" cycles="); nr_hex64(cycles[14]);
    nr_puts("\r\n");
  }
  if (counts[15]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_kv_cache_update_position_16x8x128_cap512 calls="); nr_hex64(counts[15]);
    nr_puts(" cycles="); nr_hex64(cycles[15]);
    nr_puts("\r\n");
  }
  if (counts[16]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_kv_cache_update_position_1x8x128_cap512 calls="); nr_hex64(counts[16]);
    nr_puts(" cycles="); nr_hex64(cycles[16]);
    nr_puts("\r\n");
  }
  if (counts[17]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_layout_context_8x16x128 calls="); nr_hex64(counts[17]);
    nr_puts(" cycles="); nr_hex64(cycles[17]);
    nr_puts("\r\n");
  }
  if (counts[18]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_layout_context_8x1x128 calls="); nr_hex64(counts[18]);
    nr_puts(" cycles="); nr_hex64(cycles[18]);
    nr_puts("\r\n");
  }
  if (counts[19]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_16x1024x1024 calls="); nr_hex64(counts[19]);
    nr_puts(" cycles="); nr_hex64(cycles[19]);
    nr_puts("\r\n");
  }
  if (counts[20]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_16x1024x2048 calls="); nr_hex64(counts[20]);
    nr_puts(" cycles="); nr_hex64(cycles[20]);
    nr_puts("\r\n");
  }
  if (counts[21]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_16x1024x3072 calls="); nr_hex64(counts[21]);
    nr_puts(" cycles="); nr_hex64(cycles[21]);
    nr_puts("\r\n");
  }
  if (counts[22]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_16x2048x1024 calls="); nr_hex64(counts[22]);
    nr_puts(" cycles="); nr_hex64(cycles[22]);
    nr_puts("\r\n");
  }
  if (counts[23]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_16x3072x1024 calls="); nr_hex64(counts[23]);
    nr_puts(" cycles="); nr_hex64(cycles[23]);
    nr_puts("\r\n");
  }
  if (counts[24]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_1x1024x1024 calls="); nr_hex64(counts[24]);
    nr_puts(" cycles="); nr_hex64(cycles[24]);
    nr_puts("\r\n");
  }
  if (counts[25]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_1x1024x2048 calls="); nr_hex64(counts[25]);
    nr_puts(" cycles="); nr_hex64(cycles[25]);
    nr_puts("\r\n");
  }
  if (counts[26]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_1x1024x3072 calls="); nr_hex64(counts[26]);
    nr_puts(" cycles="); nr_hex64(cycles[26]);
    nr_puts("\r\n");
  }
  if (counts[27]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_1x151936x1024 calls="); nr_hex64(counts[27]);
    nr_puts(" cycles="); nr_hex64(cycles[27]);
    nr_puts("\r\n");
  }
  if (counts[28]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_1x2048x1024 calls="); nr_hex64(counts[28]);
    nr_puts(" cycles="); nr_hex64(cycles[28]);
    nr_puts("\r\n");
  }
  if (counts[29]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_matmul_1x3072x1024 calls="); nr_hex64(counts[29]);
    nr_puts(" cycles="); nr_hex64(cycles[29]);
    nr_puts("\r\n");
  }
  if (counts[30]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_quantize_16x1024 calls="); nr_hex64(counts[30]);
    nr_puts(" cycles="); nr_hex64(cycles[30]);
    nr_puts("\r\n");
  }
  if (counts[31]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_quantize_16x2048 calls="); nr_hex64(counts[31]);
    nr_puts(" cycles="); nr_hex64(cycles[31]);
    nr_puts("\r\n");
  }
  if (counts[32]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_quantize_16x3072 calls="); nr_hex64(counts[32]);
    nr_puts(" cycles="); nr_hex64(cycles[32]);
    nr_puts("\r\n");
  }
  if (counts[33]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_quantize_1x1024 calls="); nr_hex64(counts[33]);
    nr_puts(" cycles="); nr_hex64(cycles[33]);
    nr_puts("\r\n");
  }
  if (counts[34]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_quantize_1x2048 calls="); nr_hex64(counts[34]);
    nr_puts(" cycles="); nr_hex64(cycles[34]);
    nr_puts("\r\n");
  }
  if (counts[35]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_quantize_1x3072 calls="); nr_hex64(counts[35]);
    nr_puts(" cycles="); nr_hex64(cycles[35]);
    nr_puts("\r\n");
  }
  if (counts[36]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_rmsnorm_128x128 calls="); nr_hex64(counts[36]);
    nr_puts(" cycles="); nr_hex64(cycles[36]);
    nr_puts("\r\n");
  }
  if (counts[37]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_rmsnorm_16x1024 calls="); nr_hex64(counts[37]);
    nr_puts(" cycles="); nr_hex64(cycles[37]);
    nr_puts("\r\n");
  }
  if (counts[38]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_rmsnorm_16x128 calls="); nr_hex64(counts[38]);
    nr_puts(" cycles="); nr_hex64(cycles[38]);
    nr_puts("\r\n");
  }
  if (counts[39]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_rmsnorm_1x1024 calls="); nr_hex64(counts[39]);
    nr_puts(" cycles="); nr_hex64(cycles[39]);
    nr_puts("\r\n");
  }
  if (counts[40]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_rmsnorm_256x128 calls="); nr_hex64(counts[40]);
    nr_puts(" cycles="); nr_hex64(cycles[40]);
    nr_puts("\r\n");
  }
  if (counts[41]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_rmsnorm_8x128 calls="); nr_hex64(counts[41]);
    nr_puts(" cycles="); nr_hex64(cycles[41]);
    nr_puts("\r\n");
  }
  if (counts[42]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_silu_16x3072 calls="); nr_hex64(counts[42]);
    nr_puts(" cycles="); nr_hex64(cycles[42]);
    nr_puts("\r\n");
  }
  if (counts[43]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_silu_1x3072 calls="); nr_hex64(counts[43]);
    nr_puts(" cycles="); nr_hex64(cycles[43]);
    nr_puts("\r\n");
  }
  if (counts[44]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_softmax_16x16x512 calls="); nr_hex64(counts[44]);
    nr_puts(" cycles="); nr_hex64(cycles[44]);
    nr_puts("\r\n");
  }
  if (counts[45]) {
    nr_puts("[profile] position="); nr_hex32(position);
    nr_puts(" kernel=_mlir_ciface_kernel_softmax_16x1x512 calls="); nr_hex64(counts[45]);
    nr_puts(" cycles="); nr_hex64(cycles[45]);
    nr_puts("\r\n");
  }
}
extern void __real__mlir_ciface_kernel_attention_pv_position_16x16x128x512(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2, MemRef3 *a3);
void __wrap__mlir_ciface_kernel_attention_pv_position_16x16x128x512(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2, MemRef3 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_attention_pv_position_16x16x128x512(a0, a1, a2, a3);
  ame_fence();
  cycles[0] += nr_cycles()-begin; ++counts[0];
}
extern void __real__mlir_ciface_kernel_attention_pv_position_16x1x128x512(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2, MemRef3 *a3);
void __wrap__mlir_ciface_kernel_attention_pv_position_16x1x128x512(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2, MemRef3 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_attention_pv_position_16x1x128x512(a0, a1, a2, a3);
  ame_fence();
  cycles[1] += nr_cycles()-begin; ++counts[1];
}
extern void __real__mlir_ciface_kernel_attention_qk_position_native_16x16x512x128(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2, MemRef3 *a3);
void __wrap__mlir_ciface_kernel_attention_qk_position_native_16x16x512x128(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2, MemRef3 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_attention_qk_position_native_16x16x512x128(a0, a1, a2, a3);
  ame_fence();
  cycles[2] += nr_cycles()-begin; ++counts[2];
}
extern void __real__mlir_ciface_kernel_attention_qk_position_native_16x1x512x128(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2, MemRef3 *a3);
void __wrap__mlir_ciface_kernel_attention_qk_position_native_16x1x512x128(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2, MemRef3 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_attention_qk_position_native_16x1x512x128(a0, a1, a2, a3);
  ame_fence();
  cycles[3] += nr_cycles()-begin; ++counts[3];
}
extern void __real__mlir_ciface_kernel_attention_scale_mask_position_16x16x512(MemRef3 *a0, MemRef1 *a1, MemRef3 *a2);
void __wrap__mlir_ciface_kernel_attention_scale_mask_position_16x16x512(MemRef3 *a0, MemRef1 *a1, MemRef3 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_attention_scale_mask_position_16x16x512(a0, a1, a2);
  ame_fence();
  cycles[4] += nr_cycles()-begin; ++counts[4];
}
extern void __real__mlir_ciface_kernel_attention_scale_mask_position_16x1x512(MemRef3 *a0, MemRef1 *a1, MemRef3 *a2);
void __wrap__mlir_ciface_kernel_attention_scale_mask_position_16x1x512(MemRef3 *a0, MemRef1 *a1, MemRef3 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_attention_scale_mask_position_16x1x512(a0, a1, a2);
  ame_fence();
  cycles[5] += nr_cycles()-begin; ++counts[5];
}
extern void __real__mlir_ciface_kernel_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_dequantize_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_dequantize_16x1024(a0, a1, a2, a3);
  ame_fence();
  cycles[6] += nr_cycles()-begin; ++counts[6];
}
extern void __real__mlir_ciface_kernel_dequantize_16x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_dequantize_16x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_dequantize_16x2048(a0, a1, a2, a3);
  ame_fence();
  cycles[7] += nr_cycles()-begin; ++counts[7];
}
extern void __real__mlir_ciface_kernel_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_dequantize_16x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_dequantize_16x3072(a0, a1, a2, a3);
  ame_fence();
  cycles[8] += nr_cycles()-begin; ++counts[8];
}
extern void __real__mlir_ciface_kernel_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_dequantize_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_dequantize_1x1024(a0, a1, a2, a3);
  ame_fence();
  cycles[9] += nr_cycles()-begin; ++counts[9];
}
extern void __real__mlir_ciface_kernel_dequantize_1x151936(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_dequantize_1x151936(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_dequantize_1x151936(a0, a1, a2, a3);
  ame_fence();
  cycles[10] += nr_cycles()-begin; ++counts[10];
}
extern void __real__mlir_ciface_kernel_dequantize_1x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_dequantize_1x2048(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_dequantize_1x2048(a0, a1, a2, a3);
  ame_fence();
  cycles[11] += nr_cycles()-begin; ++counts[11];
}
extern void __real__mlir_ciface_kernel_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_dequantize_1x3072(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_dequantize_1x3072(a0, a1, a2, a3);
  ame_fence();
  cycles[12] += nr_cycles()-begin; ++counts[12];
}
extern void __real__mlir_ciface_kernel_embedding_w8a8_16x1024(MemRef1 *a0, MemRef2 *a1, MemRef2 *a2, MemRef1 *a3);
void __wrap__mlir_ciface_kernel_embedding_w8a8_16x1024(MemRef1 *a0, MemRef2 *a1, MemRef2 *a2, MemRef1 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_embedding_w8a8_16x1024(a0, a1, a2, a3);
  ame_fence();
  cycles[13] += nr_cycles()-begin; ++counts[13];
}
extern void __real__mlir_ciface_kernel_embedding_w8a8_1x1024(MemRef1 *a0, MemRef2 *a1, MemRef2 *a2, MemRef1 *a3);
void __wrap__mlir_ciface_kernel_embedding_w8a8_1x1024(MemRef1 *a0, MemRef2 *a1, MemRef2 *a2, MemRef1 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_embedding_w8a8_1x1024(a0, a1, a2, a3);
  ame_fence();
  cycles[14] += nr_cycles()-begin; ++counts[14];
}
extern void __real__mlir_ciface_kernel_kv_cache_update_position_16x8x128_cap512(MemRef3 *a0, MemRef1 *a1, MemRef3 *a2);
void __wrap__mlir_ciface_kernel_kv_cache_update_position_16x8x128_cap512(MemRef3 *a0, MemRef1 *a1, MemRef3 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_kv_cache_update_position_16x8x128_cap512(a0, a1, a2);
  ame_fence();
  cycles[15] += nr_cycles()-begin; ++counts[15];
}
extern void __real__mlir_ciface_kernel_kv_cache_update_position_1x8x128_cap512(MemRef3 *a0, MemRef1 *a1, MemRef3 *a2);
void __wrap__mlir_ciface_kernel_kv_cache_update_position_1x8x128_cap512(MemRef3 *a0, MemRef1 *a1, MemRef3 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_kv_cache_update_position_1x8x128_cap512(a0, a1, a2);
  ame_fence();
  cycles[16] += nr_cycles()-begin; ++counts[16];
}
extern void __real__mlir_ciface_kernel_layout_context_8x16x128(MemRef3 *a0, MemRef3 *a1);
void __wrap__mlir_ciface_kernel_layout_context_8x16x128(MemRef3 *a0, MemRef3 *a1) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_layout_context_8x16x128(a0, a1);
  ame_fence();
  cycles[17] += nr_cycles()-begin; ++counts[17];
}
extern void __real__mlir_ciface_kernel_layout_context_8x1x128(MemRef3 *a0, MemRef3 *a1);
void __wrap__mlir_ciface_kernel_layout_context_8x1x128(MemRef3 *a0, MemRef3 *a1) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_layout_context_8x1x128(a0, a1);
  ame_fence();
  cycles[18] += nr_cycles()-begin; ++counts[18];
}
extern void __real__mlir_ciface_kernel_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_16x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_16x1024x1024(a0, a1, a2);
  ame_fence();
  cycles[19] += nr_cycles()-begin; ++counts[19];
}
extern void __real__mlir_ciface_kernel_matmul_16x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_16x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_16x1024x2048(a0, a1, a2);
  ame_fence();
  cycles[20] += nr_cycles()-begin; ++counts[20];
}
extern void __real__mlir_ciface_kernel_matmul_16x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_16x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_16x1024x3072(a0, a1, a2);
  ame_fence();
  cycles[21] += nr_cycles()-begin; ++counts[21];
}
extern void __real__mlir_ciface_kernel_matmul_16x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_16x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_16x2048x1024(a0, a1, a2);
  ame_fence();
  cycles[22] += nr_cycles()-begin; ++counts[22];
}
extern void __real__mlir_ciface_kernel_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_16x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_16x3072x1024(a0, a1, a2);
  ame_fence();
  cycles[23] += nr_cycles()-begin; ++counts[23];
}
extern void __real__mlir_ciface_kernel_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_1x1024x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_1x1024x1024(a0, a1, a2);
  ame_fence();
  cycles[24] += nr_cycles()-begin; ++counts[24];
}
extern void __real__mlir_ciface_kernel_matmul_1x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_1x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_1x1024x2048(a0, a1, a2);
  ame_fence();
  cycles[25] += nr_cycles()-begin; ++counts[25];
}
extern void __real__mlir_ciface_kernel_matmul_1x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_1x1024x3072(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_1x1024x3072(a0, a1, a2);
  ame_fence();
  cycles[26] += nr_cycles()-begin; ++counts[26];
}
extern void __real__mlir_ciface_kernel_matmul_1x151936x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_1x151936x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_1x151936x1024(a0, a1, a2);
  ame_fence();
  cycles[27] += nr_cycles()-begin; ++counts[27];
}
extern void __real__mlir_ciface_kernel_matmul_1x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_1x2048x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_1x2048x1024(a0, a1, a2);
  ame_fence();
  cycles[28] += nr_cycles()-begin; ++counts[28];
}
extern void __real__mlir_ciface_kernel_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_1x3072x1024(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_matmul_1x3072x1024(a0, a1, a2);
  ame_fence();
  cycles[29] += nr_cycles()-begin; ++counts[29];
}
extern void __real__mlir_ciface_kernel_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_kernel_quantize_16x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_quantize_16x1024(a0, a1, a2);
  ame_fence();
  cycles[30] += nr_cycles()-begin; ++counts[30];
}
extern void __real__mlir_ciface_kernel_quantize_16x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_kernel_quantize_16x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_quantize_16x2048(a0, a1, a2);
  ame_fence();
  cycles[31] += nr_cycles()-begin; ++counts[31];
}
extern void __real__mlir_ciface_kernel_quantize_16x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_kernel_quantize_16x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_quantize_16x3072(a0, a1, a2);
  ame_fence();
  cycles[32] += nr_cycles()-begin; ++counts[32];
}
extern void __real__mlir_ciface_kernel_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_kernel_quantize_1x1024(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_quantize_1x1024(a0, a1, a2);
  ame_fence();
  cycles[33] += nr_cycles()-begin; ++counts[33];
}
extern void __real__mlir_ciface_kernel_quantize_1x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_kernel_quantize_1x2048(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_quantize_1x2048(a0, a1, a2);
  ame_fence();
  cycles[34] += nr_cycles()-begin; ++counts[34];
}
extern void __real__mlir_ciface_kernel_quantize_1x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2);
void __wrap__mlir_ciface_kernel_quantize_1x3072(MemRef2 *a0, MemRef2 *a1, MemRef1 *a2) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_quantize_1x3072(a0, a1, a2);
  ame_fence();
  cycles[35] += nr_cycles()-begin; ++counts[35];
}
extern void __real__mlir_ciface_kernel_rmsnorm_128x128(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_rmsnorm_128x128(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_rmsnorm_128x128(a0, a1, a2, a3);
  ame_fence();
  cycles[36] += nr_cycles()-begin; ++counts[36];
}
extern void __real__mlir_ciface_kernel_rmsnorm_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_rmsnorm_16x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_rmsnorm_16x1024(a0, a1, a2, a3);
  ame_fence();
  cycles[37] += nr_cycles()-begin; ++counts[37];
}
extern void __real__mlir_ciface_kernel_rmsnorm_16x128(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_rmsnorm_16x128(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_rmsnorm_16x128(a0, a1, a2, a3);
  ame_fence();
  cycles[38] += nr_cycles()-begin; ++counts[38];
}
extern void __real__mlir_ciface_kernel_rmsnorm_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_rmsnorm_1x1024(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_rmsnorm_1x1024(a0, a1, a2, a3);
  ame_fence();
  cycles[39] += nr_cycles()-begin; ++counts[39];
}
extern void __real__mlir_ciface_kernel_rmsnorm_256x128(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_rmsnorm_256x128(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_rmsnorm_256x128(a0, a1, a2, a3);
  ame_fence();
  cycles[40] += nr_cycles()-begin; ++counts[40];
}
extern void __real__mlir_ciface_kernel_rmsnorm_8x128(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3);
void __wrap__mlir_ciface_kernel_rmsnorm_8x128(MemRef2 *a0, MemRef1 *a1, MemRef1 *a2, MemRef2 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_rmsnorm_8x128(a0, a1, a2, a3);
  ame_fence();
  cycles[41] += nr_cycles()-begin; ++counts[41];
}
extern void __real__mlir_ciface_kernel_silu_16x3072(MemRef2 *a0, MemRef2 *a1);
void __wrap__mlir_ciface_kernel_silu_16x3072(MemRef2 *a0, MemRef2 *a1) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_silu_16x3072(a0, a1);
  ame_fence();
  cycles[42] += nr_cycles()-begin; ++counts[42];
}
extern void __real__mlir_ciface_kernel_silu_1x3072(MemRef2 *a0, MemRef2 *a1);
void __wrap__mlir_ciface_kernel_silu_1x3072(MemRef2 *a0, MemRef2 *a1) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_silu_1x3072(a0, a1);
  ame_fence();
  cycles[43] += nr_cycles()-begin; ++counts[43];
}
extern void __real__mlir_ciface_kernel_softmax_16x16x512(MemRef3 *a0, MemRef1 *a1, MemRef1 *a2, MemRef3 *a3);
void __wrap__mlir_ciface_kernel_softmax_16x16x512(MemRef3 *a0, MemRef1 *a1, MemRef1 *a2, MemRef3 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_softmax_16x16x512(a0, a1, a2, a3);
  ame_fence();
  cycles[44] += nr_cycles()-begin; ++counts[44];
}
extern void __real__mlir_ciface_kernel_softmax_16x1x512(MemRef3 *a0, MemRef1 *a1, MemRef1 *a2, MemRef3 *a3);
void __wrap__mlir_ciface_kernel_softmax_16x1x512(MemRef3 *a0, MemRef1 *a1, MemRef1 *a2, MemRef3 *a3) {
  uint64_t begin=nr_cycles();
  __real__mlir_ciface_kernel_softmax_16x1x512(a0, a1, a2, a3);
  ame_fence();
  cycles[45] += nr_cycles()-begin; ++counts[45];
}
