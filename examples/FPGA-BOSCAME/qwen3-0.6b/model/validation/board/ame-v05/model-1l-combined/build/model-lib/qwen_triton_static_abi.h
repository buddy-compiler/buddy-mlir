#ifndef QWEN_TRITON_STATIC_ABI_H
#define QWEN_TRITON_STATIC_ABI_H
#include <stdint.h>
/* Generated static-shape ABI. Graph lowering must validate size/stride/alias first. */
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { void *allocated, *aligned; int64_t offset, sizes[1], strides[1]; } QwenMemRef1;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[2], strides[2]; } QwenMemRef2;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[3], strides[3]; } QwenMemRef3;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[4], strides[4]; } QwenMemRef4;
void _mlir_ciface_kernel_attention_pv_position_16x16x128x512(QwenMemRef3 *, QwenMemRef3 *, QwenMemRef1 *, QwenMemRef3 *);
void _mlir_ciface_kernel_attention_pv_position_16x1x128x512(QwenMemRef3 *, QwenMemRef3 *, QwenMemRef1 *, QwenMemRef3 *);
void _mlir_ciface_kernel_attention_qk_position_native_16x16x512x128(QwenMemRef3 *, QwenMemRef3 *, QwenMemRef1 *, QwenMemRef3 *);
void _mlir_ciface_kernel_attention_qk_position_native_16x1x512x128(QwenMemRef3 *, QwenMemRef3 *, QwenMemRef1 *, QwenMemRef3 *);
void _mlir_ciface_kernel_attention_scale_mask_position_16x16x512(QwenMemRef3 *, QwenMemRef1 *, QwenMemRef3 *);
void _mlir_ciface_kernel_attention_scale_mask_position_16x1x512(QwenMemRef3 *, QwenMemRef1 *, QwenMemRef3 *);
void _mlir_ciface_kernel_dequantize_16x1024(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_dequantize_16x2048(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_dequantize_16x3072(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_dequantize_1x1024(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_dequantize_1x151936(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_dequantize_1x2048(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_dequantize_1x3072(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_embedding_w8a8_16x1024(QwenMemRef1 *, QwenMemRef2 *, QwenMemRef2 *, QwenMemRef1 *);
void _mlir_ciface_kernel_embedding_w8a8_1x1024(QwenMemRef1 *, QwenMemRef2 *, QwenMemRef2 *, QwenMemRef1 *);
void _mlir_ciface_kernel_kv_cache_update_position_16x8x128_cap512(QwenMemRef3 *, QwenMemRef1 *, QwenMemRef3 *);
void _mlir_ciface_kernel_kv_cache_update_position_1x8x128_cap512(QwenMemRef3 *, QwenMemRef1 *, QwenMemRef3 *);
void _mlir_ciface_kernel_layout_context_8x16x128(QwenMemRef3 *, QwenMemRef3 *);
void _mlir_ciface_kernel_layout_context_8x1x128(QwenMemRef3 *, QwenMemRef3 *);
void _mlir_ciface_kernel_matmul_16x1024x1024(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_16x1024x2048(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_16x1024x3072(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_16x2048x1024(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_16x3072x1024(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_1x1024x1024(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_1x1024x2048(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_1x1024x3072(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_1x151936x1024(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_1x2048x1024(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_matmul_1x3072x1024(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_quantize_16x1024(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef1 *);
void _mlir_ciface_kernel_quantize_16x2048(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef1 *);
void _mlir_ciface_kernel_quantize_16x3072(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef1 *);
void _mlir_ciface_kernel_quantize_1x1024(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef1 *);
void _mlir_ciface_kernel_quantize_1x2048(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef1 *);
void _mlir_ciface_kernel_quantize_1x3072(QwenMemRef2 *, QwenMemRef2 *, QwenMemRef1 *);
void _mlir_ciface_kernel_rmsnorm_128x128(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_rmsnorm_16x1024(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_rmsnorm_16x128(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_rmsnorm_1x1024(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_rmsnorm_256x128(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_rmsnorm_8x128(QwenMemRef2 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef2 *);
void _mlir_ciface_kernel_silu_16x3072(QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_silu_1x3072(QwenMemRef2 *, QwenMemRef2 *);
void _mlir_ciface_kernel_softmax_16x16x512(QwenMemRef3 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef3 *);
void _mlir_ciface_kernel_softmax_16x1x512(QwenMemRef3 *, QwenMemRef1 *, QwenMemRef1 *, QwenMemRef3 *);
#ifdef __cplusplus
}
#endif
#endif
