#include "Interface.h"
#include "Profiler.h"
#include <cstdint>

using namespace buddy::runtime;

void _mlir_ciface_timingStart(uint64_t ptr) {
  Profiler::getTimeManager().timingStart(reinterpret_cast<TimeEvent *>(ptr));
}

void _mlir_ciface_timingEnd(uint64_t ptr) {
  Profiler::getTimeManager().timingEnd(reinterpret_cast<TimeEvent *>(ptr));
}