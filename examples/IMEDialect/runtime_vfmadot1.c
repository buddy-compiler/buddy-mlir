#include <stdio.h>

void print_header() {
  printf("=== vfmadot1 (fp16 x fp16, fixed slide=1) ===");
  printf("\n\nMatrix A (8x4, fp16, sliding source):\n");
  printf("  [1.0, 2.0, 3.0, 4.0]   row 0\n");
  printf("  [2.0, 3.0, 4.0, 5.0]   row 1 <- slide=1 starts here\n");
  printf("  [3.0, 4.0, 5.0, 6.0]   row 2\n");
  printf("  [4.0, 5.0, 6.0, 7.0]   row 3\n");
  printf("  [5.0, 6.0, 7.0, 8.0]   row 4\n");
  printf("  [6.0, 7.0, 8.0, 9.0]   row 5\n");
  printf("  [7.0, 8.0, 9.0, 10.0]  row 6\n");
  printf("  [8.0, 9.0, 10.0, 11.0] row 7\n");
  printf("\nMatrix B (4x4, fp16, packed):\n");
  printf("  [1.0, 1.0, 1.0, 1.0]\n");
  printf("  [1.0, 1.0, 1.0, 1.0]\n");
  printf("  [1.0, 1.0, 1.0, 1.0]\n");
  printf("  [1.0, 1.0, 1.0, 1.0]\n");
  printf("\nFixed slide: 1\n");
  printf("Rows used after slide: [1,2,3,4] -> sums [14,18,22,26]\n");
  printf("\nResult matrix C (4x4, fp16):\n");
  printf("Expected: [[14,14,14,14], [18,18,18,18], [22,22,22,22], "
         "[26,26,26,26]]\n");
}

void print_row_f16(int row, float v0, float v1, float v2, float v3) {
  printf("Row %d: [%.1f, %.1f, %.1f, %.1f]\n", row, v0, v1, v2, v3);
}
