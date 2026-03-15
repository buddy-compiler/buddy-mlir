/**
 * Runtime test for matmul with boundary handling
 * 
 * This test verifies the correctness of linalg.matmul -> IME lowering
 * for matrices with dimensions not aligned to IME tile sizes.
 * 
 * Test case: C[7x5] = A[7x10] * B[10x5]
 * For int8: TILE_M=4, TILE_K=8, TILE_N=4
 * - M=7: 1 full tile (4) + 3 remaining
 * - N=5: 1 full tile (4) + 1 remaining  
 * - K=10: 1 full tile (8) + 2 remaining
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Matrix dimensions
#define M 7
#define K 10
#define N 5

// Test matrices
int8_t A[M][K] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    {3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    {5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
};

int8_t B[K][N] = {
    {1, 0, 0, 0, 0},
    {0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 0, 1},
    {1, 1, 1, 1, 1},
    {2, 2, 2, 2, 2},
    {3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4},
    {5, 5, 5, 5, 5}
};

int32_t C[M][N];
int32_t C_expected[M][N];

// External function from compiled MLIR
// The function uses unpacked memref arguments:
// (allocated, aligned, offset, size0, size1, stride0, stride1) for each memref
extern void matmul_boundary(
    int8_t *A_alloc, int8_t *A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    int8_t *B_alloc, int8_t *B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    int32_t *C_alloc, int32_t *C_aligned, int64_t C_offset,
    int64_t C_size0, int64_t C_size1, int64_t C_stride0, int64_t C_stride1);

// Reference implementation for verification
void reference_matmul() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_expected[i][j] = 0;
            for (int k = 0; k < K; k++) {
                C_expected[i][j] += (int32_t)A[i][k] * (int32_t)B[k][j];
            }
        }
    }
}

void print_matrix_i32(const char *name, int32_t *mat, int rows, int cols) {
    printf("%s [%dx%d]:\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("%4d", mat[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

int verify_result() {
    int errors = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i][j] != C_expected[i][j]) {
                printf("ERROR at C[%d][%d]: got %d, expected %d\n",
                       i, j, C[i][j], C_expected[i][j]);
                errors++;
            }
        }
    }
    return errors;
}

int main() {
    printf("=== Matmul Boundary Test ===\n");
    printf("Matrix dimensions: A[%dx%d] * B[%dx%d] = C[%dx%d]\n",
           M, K, K, N, M, N);
    printf("IME int8 tile sizes: TILE_M=4, TILE_K=8, TILE_N=4\n");
    printf("Boundary cases:\n");
    printf("  M=%d: %d full tiles + %d remaining\n", M, M/4, M%4);
    printf("  K=%d: %d full tiles + %d remaining\n", K, K/8, K%8);
    printf("  N=%d: %d full tiles + %d remaining\n\n", N, N/4, N%4);

    // Initialize output matrix
    memset(C, 0, sizeof(C));

    // Compute reference result
    reference_matmul();
    
    // Call MLIR-generated function with unpacked memref arguments
    // Each memref: (allocated, aligned, offset, size0, size1, stride0, stride1)
    matmul_boundary(
        (int8_t*)A, (int8_t*)A, 0, M, K, K, 1,   // A[7x10]
        (int8_t*)B, (int8_t*)B, 0, K, N, N, 1,   // B[10x5]
        (int32_t*)C, (int32_t*)C, 0, M, N, N, 1  // C[7x5]
    );

    // Print results
    print_matrix_i32("Result C", (int32_t*)C, M, N);
    printf("\n");
    print_matrix_i32("Expected C", (int32_t*)C_expected, M, N);
    printf("\n");

    // Verify
    int errors = verify_result();
    if (errors == 0) {
        printf("PASS: All results match!\n");
        return 0;
    } else {
        printf("FAIL: %d errors found\n", errors);
        return 1;
    }
}
