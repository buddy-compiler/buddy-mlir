// ===- onednn_ops.h
// ------------------------------------------------------------
//
// C Interface Wrapper for oneDNN Operations
// Provides functions callable from MLIR-generated code
//
// ===---------------------------------------------------------------------------

#ifndef BUDDY_ONEDNN_OPS_H
#define BUDDY_ONEDNN_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MLIR MemRef Structure Definitions
// ============================================================================

/**
 * MLIR MemRef Descriptor (2D)
 * Corresponds to MLIR type: !llvm.struct<(ptr, ptr, i64, array<2 x i64>,
 * array<2 x i64>)>
 */
typedef struct {
  float *allocated;   // Allocated memory pointer
  float *aligned;     // Aligned data pointer (actual data)
  int64_t offset;     // Offset
  int64_t sizes[2];   // Dimension sizes [dim0, dim1]
  int64_t strides[2]; // Strides [stride0, stride1]
} MemRefDescriptor2D_f32;

// ============================================================================
// MatMul Operations
// ============================================================================

/**
 * 2D Matrix Multiplication: C = A * B
 *
 * @param A Input matrix A data pointer
 * @param A_shape A's shape array [M, K]
 * @param A_rank A's rank (should be 2)
 * @param B Input matrix B data pointer
 * @param B_shape B's shape array [K, N]
 * @param B_rank B's rank (should be 2)
 * @param C Output matrix C data pointer
 * @param C_shape C's shape array [M, N]
 * @param C_rank C's rank (should be 2)
 */
void onednn_matmul_2d_f32(float *A, int64_t *A_shape, int64_t A_rank, float *B,
                          int64_t *B_shape, int64_t B_rank, float *C,
                          int64_t *C_shape, int64_t C_rank);

/**
 * 3D Batch Matrix Multiplication: C[b] = A[b] * B[b]
 *
 * @param A Input tensor A data pointer
 * @param A_shape A's shape array [batch, M, K]
 * @param A_rank A's rank (should be 3)
 * @param B Input tensor B data pointer
 * @param B_shape B's shape array [batch, K, N] or [K, N]
 * @param B_rank B's rank (2 or 3)
 * @param C Output tensor C data pointer
 * @param C_shape C's shape array [batch, M, N]
 * @param C_rank C's rank (should be 3)
 */
void onednn_matmul_3d_f32(float *A, int64_t *A_shape, int64_t A_rank, float *B,
                          int64_t *B_shape, int64_t B_rank, float *C,
                          int64_t *C_shape, int64_t C_rank);

/**
 * Generic Matrix Multiplication (auto-handles 2D/3D)
 */
void onednn_matmul_f32(float *A, int64_t *A_shape, int64_t A_rank, float *B,
                       int64_t *B_shape, int64_t B_rank, float *C,
                       int64_t *C_shape, int64_t C_rank);

// ============================================================================
// MLIR C Interface (for MLIR-generated code)
// ============================================================================

/**
 * MLIR C Interface: 2D Matrix Multiplication
 *
 * This is the interface expected by MLIR-generated code.
 * Function name must be _mlir_ciface_onednn_matmul_f32
 *
 * @param result Output MemRef (C = A * B)
 * @param A Input MemRef A
 * @param B Input MemRef B
 */
void _mlir_ciface_onednn_matmul_f32(MemRefDescriptor2D_f32 *result,
                                    MemRefDescriptor2D_f32 *A,
                                    MemRefDescriptor2D_f32 *B);

#ifdef __cplusplus
}
#endif

#endif // BUDDY_ONEDNN_OPS_H
