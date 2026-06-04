//===- MatmulTorq.cpp -----------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// TORQ-Tile implementation of buddy_matmul_f32 (RVV GEMM micro-kernel).
//
//===----------------------------------------------------------------------===//

#if !defined(__riscv) || !defined(__riscv_v)
#error buddy_matmul_torq must be compiled for RISC-V with the Vector extension.
#endif

#include "buddy/runtime/Matmul.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "tqt/gemm/gemm_1xnbias_clamp_f32_f32_f32_ukernel.h"

extern "C" {
size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(void);
size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(void);
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(size_t m_idx,
                                                                 size_t k_idx,
                                                                 size_t lda);
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(size_t n_idx,
                                                                 size_t k_idx,
                                                                 size_t ldb);
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(size_t m_idx,
                                                                 size_t n_idx,
                                                                 size_t ldc);
size_t
tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(size_t m_idx,
                                                                 size_t n_idx,
                                                                 size_t ldd);
size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(size_t m,
                                                               size_t n);
void tqt_run_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc,
    void *D, size_t ldd, const void *bias, float clamp_min, float clamp_max);
}

namespace {

struct RawMemRef2D {
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
};

const tqt_gemm_1xnbias_clamp_f32_f32_f32_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    tqt_run_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv};

void torq_gemm_matmul_only(size_t M, size_t N, size_t K, const float *A,
                           size_t lda, const float *B, size_t ldb, float *D,
                           size_t ldd) {
  std::vector<float> C_zeros(M * N, 0.0f);
  std::vector<float> bias_zeros(N, 0.0f);
  const float clamp_min = -1e6f;
  const float clamp_max = 1e6f;

  const size_t m_step = ukernel.get_m_step();
  const size_t n_step = ukernel.get_n_step();

  for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
    for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
      const size_t actual_m = std::min(M - m_idx, m_step);
      const size_t actual_n = std::min(N - n_idx, n_step);

      const uint8_t *a_ptr =
          (const uint8_t *)A + ukernel.get_a_offset(m_idx, 0, lda);
      const uint8_t *b_ptr =
          (const uint8_t *)B + ukernel.get_b_offset(n_idx, 0, ldb);
      const uint8_t *c_ptr = (const uint8_t *)C_zeros.data() +
                             ukernel.get_c_offset(m_idx, n_idx, N);
      const uint8_t *bias_ptr =
          (const uint8_t *)bias_zeros.data() + ukernel.get_bias_offset(n_idx);
      uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, ldd);

      ukernel.run_gemm(actual_m, actual_n, K, a_ptr, lda, 0, b_ptr, ldb, 0,
                       c_ptr, N, d_ptr, ldd, bias_ptr, clamp_min, clamp_max);
    }
  }
}

} // namespace

extern "C" {

void _mlir_ciface_buddy_matmul_f32(MemRef<float, 2> *result,
                                   MemRef<float, 2> *A, MemRef<float, 2> *B) {
  const intptr_t *sA = A->getSizes();
  const intptr_t *sB = B->getSizes();
  const int64_t M = sA[0];
  const int64_t K = sA[1];
  const int64_t N = sB[1];

  const intptr_t *stA = A->getStrides();
  const intptr_t *stB = B->getStrides();
  const float *a_raw = A->getData();
  const float *b_raw = B->getData();

  std::vector<float> a_buf;
  std::vector<float> b_buf;
  const float *a_ptr = a_raw;
  const float *b_ptr = b_raw;

  if (stA[1] != 1 || stA[0] != K) {
    a_buf.assign(static_cast<size_t>(M * K), 0.0f);
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < K; ++j)
        a_buf[static_cast<size_t>(i * K + j)] = a_raw[i * stA[0] + j * stA[1]];
    a_ptr = a_buf.data();
  }
  if (stB[1] != 1 || stB[0] != N) {
    b_buf.assign(static_cast<size_t>(K * N), 0.0f);
    for (int64_t i = 0; i < K; ++i)
      for (int64_t j = 0; j < N; ++j)
        b_buf[static_cast<size_t>(i * N + j)] = b_raw[i * stB[0] + j * stB[1]];
    b_ptr = b_buf.data();
  }

  float *d_data = (float *)malloc(static_cast<size_t>(M * N) * sizeof(float));
  if (!d_data)
    return;

  torq_gemm_matmul_only(static_cast<size_t>(M), static_cast<size_t>(N),
                        static_cast<size_t>(K), a_ptr, static_cast<size_t>(K),
                        b_ptr, static_cast<size_t>(N), d_data,
                        static_cast<size_t>(N));

  auto *out = reinterpret_cast<RawMemRef2D *>(result);
  out->allocated = d_data;
  out->aligned = d_data;
  out->offset = 0;
  out->sizes[0] = M;
  out->sizes[1] = N;
  out->strides[0] = N;
  out->strides[1] = 1;
}

} // extern "C"
