#include "RVProfRuntime.h"

extern void _mlir_ciface_rvprof_test(void);

int main(void) {
  __rvprof_init();
  _mlir_ciface_rvprof_test();
  __rvprof_dump("rvprof_trace.json");
  return 0;
}
