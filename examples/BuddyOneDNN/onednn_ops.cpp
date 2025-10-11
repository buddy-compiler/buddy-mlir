// ===- onednn_ops.cpp
// ----------------------------------------------------------
//
// oneDNN Operations Implementation
//
// ===---------------------------------------------------------------------------

#include "onednn_ops.h"
#include <iostream>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

using namespace dnnl;

// Global engine and stream (avoid repeated creation)
static engine &get_cpu_engine() {
  static engine eng(engine::kind::cpu, 0);
  return eng;
}

static stream &get_stream() {
  static stream s(get_cpu_engine());
  return s;
}

extern "C" {

// ============================================================================
// 2D MatMul
// ============================================================================

void onednn_matmul_2d_f32(float *A, int64_t *A_shape, int64_t A_rank, float *B,
                          int64_t *B_shape, int64_t B_rank, float *C,
                          int64_t *C_shape, int64_t C_rank) {
  try {
    auto eng = get_cpu_engine();
    auto s = get_stream();

    // Parse shapes: A[M, K], B[K, N], C[M, N]
    int64_t M = A_shape[0];
    int64_t K = A_shape[1];
    int64_t N = B_shape[1];

    // Create memory descriptors
    memory::dims a_dims = {M, K};
    memory::dims b_dims = {K, N};
    memory::dims c_dims = {M, N};

    auto a_md =
        memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
    auto b_md =
        memory::desc(b_dims, memory::data_type::f32, memory::format_tag::ab);
    auto c_md =
        memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);

    // Create memory objects
    auto a_mem = memory(a_md, eng, A);
    auto b_mem = memory(b_md, eng, B);
    auto c_mem = memory(c_md, eng, C);

    // Create matmul primitive
    auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);
    auto matmul_prim = matmul(matmul_pd);

    // Execute
    matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                            {DNNL_ARG_WEIGHTS, b_mem},
                            {DNNL_ARG_DST, c_mem}});

    s.wait();

  } catch (const std::exception &e) {
    std::cerr << "[oneDNN] matmul_2d error: " << e.what() << std::endl;
  }
}

// ============================================================================
// 3D MatMul
// ============================================================================

void onednn_matmul_3d_f32(float *A, int64_t *A_shape, int64_t A_rank, float *B,
                          int64_t *B_shape, int64_t B_rank, float *C,
                          int64_t *C_shape, int64_t C_rank) {
  try {
    auto eng = get_cpu_engine();
    auto s = get_stream();

    // Parse shapes
    int64_t batch = A_shape[0];
    int64_t M = A_shape[1];
    int64_t K = A_shape[2];

    // B can be [batch, K, N] or [K, N]
    int64_t N;
    memory::dims b_dims;

    if (B_rank == 3) {
      // B is 3D: [batch, K, N]
      N = B_shape[2];
      b_dims = {batch, K, N};
    } else {
      // B is 2D: [K, N] - needs broadcasting
      N = B_shape[1];
      b_dims = {K, N};
    }

    // Create memory descriptors
    memory::dims a_dims = {batch, M, K};
    memory::dims c_dims = {batch, M, N};

    auto a_md =
        memory::desc(a_dims, memory::data_type::f32, memory::format_tag::abc);
    auto c_md =
        memory::desc(c_dims, memory::data_type::f32, memory::format_tag::abc);

    // Create memory objects
    auto a_mem = memory(a_md, eng, A);
    auto c_mem = memory(c_md, eng, C);

    if (B_rank == 3) {
      // 3D batch matrix multiplication
      auto b_md =
          memory::desc(b_dims, memory::data_type::f32, memory::format_tag::abc);
      auto b_mem = memory(b_md, eng, B);

      auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);
      auto matmul_prim = matmul(matmul_pd);

      matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                              {DNNL_ARG_WEIGHTS, b_mem},
                              {DNNL_ARG_DST, c_mem}});
    } else {
      // B is 2D, need to execute for each batch
      auto b_md =
          memory::desc(b_dims, memory::data_type::f32, memory::format_tag::ab);
      auto b_mem = memory(b_md, eng, B);

      // Execute matmul for each batch
      for (int64_t b = 0; b < batch; b++) {
        memory::dims a_batch_dims = {M, K};
        memory::dims c_batch_dims = {M, N};

        auto a_batch_md = memory::desc(a_batch_dims, memory::data_type::f32,
                                       memory::format_tag::ab);
        auto c_batch_md = memory::desc(c_batch_dims, memory::data_type::f32,
                                       memory::format_tag::ab);

        // 指向当前batch的数据
        float *a_batch_ptr = A + b * M * K;
        float *c_batch_ptr = C + b * M * N;

        auto a_batch_mem = memory(a_batch_md, eng, a_batch_ptr);
        auto c_batch_mem = memory(c_batch_md, eng, c_batch_ptr);

        auto matmul_pd =
            matmul::primitive_desc(eng, a_batch_md, b_md, c_batch_md);
        auto matmul_prim = matmul(matmul_pd);

        matmul_prim.execute(s, {{DNNL_ARG_SRC, a_batch_mem},
                                {DNNL_ARG_WEIGHTS, b_mem},
                                {DNNL_ARG_DST, c_batch_mem}});
      }
    }

    s.wait();

  } catch (const std::exception &e) {
    std::cerr << "[oneDNN] matmul_3d error: " << e.what() << std::endl;
  }
}

// ============================================================================
// Generic MatMul (auto-dispatch to 2D or 3D)
// ============================================================================

void onednn_matmul_f32(float *A, int64_t *A_shape, int64_t A_rank, float *B,
                       int64_t *B_shape, int64_t B_rank, float *C,
                       int64_t *C_shape, int64_t C_rank) {
  if (A_rank == 2 && B_rank == 2 && C_rank == 2) {
    // 2D matrix multiplication
    onednn_matmul_2d_f32(A, A_shape, A_rank, B, B_shape, B_rank, C, C_shape,
                         C_rank);
  } else if (A_rank == 3 && C_rank == 3) {
    // 3D batch matrix multiplication
    onednn_matmul_3d_f32(A, A_shape, A_rank, B, B_shape, B_rank, C, C_shape,
                         C_rank);
  } else {
    std::cerr << "[oneDNN] Unsupported matmul dimensions: "
              << "A_rank=" << A_rank << ", B_rank=" << B_rank
              << ", C_rank=" << C_rank << std::endl;
  }
}

// ============================================================================
// MLIR C Interface Implementation
// ============================================================================

void _mlir_ciface_onednn_matmul_f32(MemRefDescriptor2D_f32 *result,
                                    MemRefDescriptor2D_f32 *A,
                                    MemRefDescriptor2D_f32 *B) {
  try {
    auto eng = get_cpu_engine();
    auto s = get_stream();

    // 1. Extract dimension information
    int64_t M = A->sizes[0]; // Number of rows in A
    int64_t K = A->sizes[1]; // Number of columns in A = rows in B
    int64_t N = B->sizes[1]; // Number of columns in B

    // 2. Get data pointers (use aligned pointer, this is the actual data)
    float *a_data = A->aligned;
    float *b_data = B->aligned;

    // Check pointer validity
    if (!a_data || !b_data) {
      std::cerr << "[oneDNN] Error: null input pointers\n";
      return;
    }

    // MLIR passes result as uninitialized stack memory, we need to allocate and
    // fill it
    float *c_data = (float *)malloc(M * N * sizeof(float));
    if (!c_data) {
      std::cerr << "[oneDNN] Error: failed to allocate result buffer\n";
      return;
    }

    // Fill result MemRef descriptor
    result->allocated = c_data;
    result->aligned = c_data;
    result->offset = 0;
    result->sizes[0] = M;
    result->sizes[1] = N;
    result->strides[0] = N;
    result->strides[1] = 1;

    // 3. Create oneDNN memory descriptors
    memory::dims a_dims = {M, K};
    memory::dims b_dims = {K, N};
    memory::dims c_dims = {M, N};

    // Consider strides (MLIR may pass non-contiguous memory)
    // strides[0] = row stride, strides[1] = column stride
    memory::dims a_strides = {A->strides[0], A->strides[1]};
    memory::dims b_strides = {B->strides[0], B->strides[1]};
    memory::dims c_strides = {result->strides[0], result->strides[1]};

    // Create memory descriptors (with strides)
    auto a_md = memory::desc(a_dims, memory::data_type::f32, a_strides);
    auto b_md = memory::desc(b_dims, memory::data_type::f32, b_strides);
    auto c_md = memory::desc(c_dims, memory::data_type::f32, c_strides);

    // 4. Create memory objects (wrapping existing data)
    auto a_mem = memory(a_md, eng, a_data);
    auto b_mem = memory(b_md, eng, b_data);
    auto c_mem = memory(c_md, eng, c_data);

    // 5. Create matmul primitive descriptor
    auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);

    // 6. Create matmul primitive
    auto matmul_prim = matmul(matmul_pd);

    // 7. Execute matmul
    matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                            {DNNL_ARG_WEIGHTS, b_mem},
                            {DNNL_ARG_DST, c_mem}});

    // 8. Wait for completion
    s.wait();

  } catch (const std::exception &e) {
    std::cerr << "[oneDNN MLIR Interface] Error: " << e.what() << std::endl;
  }
}
}
