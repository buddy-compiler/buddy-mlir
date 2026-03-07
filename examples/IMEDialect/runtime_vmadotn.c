#include <stdio.h>

void print_header() {
  printf("=== vmadotn (signed x signed, dynamic slide) ===");
  printf("\n\nMatrix A (8x8, int8, signed, sliding source):\n");
  printf("  [1, 2, 3, 4, 5, 6, 7, 8]      row 0\n");
  printf("  [2, 3, 4, 5, 6, 7, 8, 9]      row 1 <- slide=1 starts here\n");
  printf("  [3, 4, 5, 6, 7, 8, 9, 10]     row 2\n");
  printf("  [4, 5, 6, 7, 8, 9, 10, 11]    row 3\n");
  printf("  [5, 6, 7, 8, 9, 10, 11, 12]   row 4\n");
  printf("  [6, 7, 8, 9, 10, 11, 12, 13]  row 5\n");
  printf("  [7, 8, 9, 10, 11, 12, 13, 14] row 6\n");
  printf("  [8, 9, 10, 11, 12, 13, 14, 15] row 7\n");
  printf("\nMatrix B (4x8, int8, signed, packed):\n");
  printf("  [1, 1, 1, 1, 1, 1, 1, 1]\n");
  printf("  [1, 1, 1, 1, 1, 1, 1, 1]\n");
  printf("  [1, 1, 1, 1, 1, 1, 1, 1]\n");
  printf("  [1, 1, 1, 1, 1, 1, 1, 1]\n");
  printf("\nSlide parameter: 1\n");
  printf("Rows used after slide: [1,2,3,4] -> sums [44,52,60,68]\n");
  printf("\nResult matrix C (4x4, int32):\n");
  printf("Expected: [[44,44,44,44], [52,52,52,52], [60,60,60,60], "
         "[68,68,68,68]]\n");
}

void print_row(int row, int v0, int v1, int v2, int v3) {
  printf("Row %d: [%d, %d, %d, %d]\n", row, v0, v1, v2, v3);
}
