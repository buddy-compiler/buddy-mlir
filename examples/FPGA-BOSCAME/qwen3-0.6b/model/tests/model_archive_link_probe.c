/* Relocatable-link evidence only. Never flashed and never used as inference.
 *
 * Calls every entry point of the model archive with the descriptor ranks the
 * generated ABI header declares. A duplicate or missing symbol, a rank mismatch
 * or an accidental test/runtime member in the archive fails this link, which is
 * what makes the archive's advertised ABI checkable rather than asserted.
 */
#include "qwen_triton_static_abi.h"

void qwen_model_archive_link_probe(
    QwenMemRef3 *m3a, QwenMemRef3 *m3b, QwenMemRef3 *m3c,
    QwenMemRef1 *m1, QwenMemRef2 *m2) {
  (void)m2;
  /* attention QK/PV: A[3], B[3], C[3] */
  _mlir_ciface_kernel_attention_qk_16x16x512x128(m3a, m3b, m3c);
  _mlir_ciface_kernel_attention_qk_16x1x512x128(m3a, m3b, m3c);
  _mlir_ciface_kernel_attention_pv_16x16x128x512(m3a, m3b, m3c);
  _mlir_ciface_kernel_attention_pv_16x1x128x512(m3a, m3b, m3c);
  /* scale+mask with a runtime position vector */
  _mlir_ciface_kernel_attention_scale_mask_position_16x16x512(m3a, m1, m3c);
  _mlir_ciface_kernel_attention_scale_mask_position_16x1x512(m3a, m1, m3c);
  /* softmax with maxima/sums side outputs */
  _mlir_ciface_kernel_softmax_16x16x512(m3a, m1, m1, m3c);
  _mlir_ciface_kernel_softmax_16x1x512(m3a, m1, m1, m3c);
  /* KV write at a runtime slot */
  _mlir_ciface_kernel_kv_cache_update_position_16x8x128_cap512(m3a, m1, m3c);
  _mlir_ciface_kernel_kv_cache_update_position_1x8x128_cap512(m3a, m1, m3c);
  /* GQA head expansion */
  _mlir_ciface_kernel_gqa_repeat_8x512x128_to_16x512x128(m3a, m3c);
}
