/* Relocatable-link evidence only. Never flashed or used as inference. */
#include "qwen_triton_static_abi.h"
void qwen_archive_link_probe(QwenMemRef2 *a, QwenMemRef2 *b, QwenMemRef2 *c) {
  _mlir_ciface_kernel_add_1x1024(a, b, c);
  _mlir_ciface_kernel_matmul_3x19x70(a, b, c);
}
