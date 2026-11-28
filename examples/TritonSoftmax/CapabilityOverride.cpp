// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Only preloaded into an isolated negative-test process, never system-wide.
#include <sys/auxv.h>
#include <sys/prctl.h>
#if TEST_CAP_MODE == 1
extern "C" unsigned long getauxval(unsigned long) noexcept { return 0; }
#else
extern "C" int prctl(int, ...) noexcept { return PR_RISCV_V_VSTATE_CTRL_OFF; }
#endif
