#include "Interface.h"
#include "TimeManager.h"
#include <cstdint>

using namespace buddy::runtime;

void _mlir_ciface_timingStart(uint64_t ptr) {
  TimeManager::timingStart(reinterpret_cast<TimeEvent *>(ptr));
}

void _mlir_ciface_timingEnd(uint64_t ptr) {
  TimeManager::timingEnd(reinterpret_cast<TimeEvent *>(ptr));
}