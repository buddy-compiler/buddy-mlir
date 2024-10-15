#include "Interface.h"
#include "TimeManager.h"

using namespace buddy::runtime;

void _mlir_ciface_timingStart() { TimeManager::timingStart(); }

void _mlir_ciface_timingEnd() { TimeManager::timingEnd(); }