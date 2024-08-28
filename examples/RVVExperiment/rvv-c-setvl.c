#include <riscv_vector.h>
#include <stdio.h>

int main() {
  int avl = 70;
  int vl = __riscv_vsetvl_e32m2(avl);
  printf("vl: %d\n", vl);

  return 0;
}
