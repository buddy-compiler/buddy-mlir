#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sys/time.h>

extern "C" double _mlir_ciface_rtclock() {
  struct timeval tp;
  int stat = gettimeofday(&tp, nullptr);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
  // return rtclock();
}

extern "C" void _mlir_ciface_printF64(double d) { printF64(d); }

extern "C" void _mlir_ciface_printNewline() { printNewline(); }