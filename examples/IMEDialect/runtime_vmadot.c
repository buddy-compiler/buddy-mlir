#include <stdio.h>

void print_header() {
  printf("=== vmadot (signed x signed) ===");
  printf("\n\nMatrix A (4x8, int8, signed):\n");
  printf("  [-1, -2, -3, -4, -5, -6, -7, -8]\n");
  printf("  [-1, -2, -3, -4, -5, -6, -7, -8]\n");
  printf("  [-1, -2, -3, -4, -5, -6, -7, -8]\n");
  printf("  [-1, -2, -3, -4, -5, -6, -7, -8]\n");
  printf("\nMatrix B (4x8, int8, signed, packed):\n");
  printf("  [-1, -1, -1, -1, -1, -1, -1, -1]\n");
  printf("  [-2, -2, -2, -2, -2, -2, -2, -2]\n");
  printf("  [-1, -1, -1, -1, -1, -1, -1, -1]\n");
  printf("  [-2, -2, -2, -2, -2, -2, -2, -2]\n");
  printf("\nResult matrix C (4x4, int32):\n");
  printf("Expected: [[36,72,36,72], [36,72,36,72], [36,72,36,72], "
         "[36,72,36,72]]\n");
}

void print_row(int row, int v0, int v1, int v2, int v3) {
  printf("Row %d: [%d, %d, %d, %d]\n", row, v0, v1, v2, v3);
}
