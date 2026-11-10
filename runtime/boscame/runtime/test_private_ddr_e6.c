//===- test_private_ddr_e6.c - Private DDR tests on NH/E6 platform --------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// Exercise the production E6 branch with modeled CSR and linker context.
#include <assert.h>
#include <stdint.h>

static uintptr_t incoming_mie, model_status, model_sp;
static uintptr_t model_e6_bottom, model_e6_top, model_nh_bottom, model_nh_top;
static unsigned saves, restores, reenter;
static void model_reset(void);
static void model_shared(void);
static uintptr_t model_irq_save(void) {
  uintptr_t saved = model_status & 8u;
  model_status &= ~(uintptr_t)8u;
  ++saves;
  return saved;
}
static void model_irq_restore(uintptr_t saved);

#define BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6 1
#define BOSC_AME_PRIVATE_DDR_E6_STACK_BOTTOM model_e6_bottom
#define BOSC_AME_PRIVATE_DDR_E6_STACK_TOP model_e6_top
#define BOSC_AME_PRIVATE_DDR_NH_STACK_BOTTOM model_nh_bottom
#define BOSC_AME_PRIVATE_DDR_NH_STACK_TOP model_nh_top
#define BOSC_AME_PRIVATE_DDR_STACK_POINTER() model_sp
#define BOSC_AME_PRIVATE_DDR_IRQ_SAVE() model_irq_save()
#define BOSC_AME_PRIVATE_DDR_IRQ_RESTORE(saved) model_irq_restore(saved)
#define BOSC_AME_PRIVATE_DDR_TEST_RESET() model_reset()
#define BOSC_AME_PRIVATE_DDR_TEST_SHARED() model_shared()
#define main generic_frame_cases
#include "test_private_ddr.c"
#undef main

static void model_reset(void) {
  model_status = UINT64_C(0x0601e600) | incoming_mie;
  model_sp = 0x18000;
  model_e6_bottom = 0x10000;
  model_e6_top = 0x20000;
  model_nh_bottom = 0x30000;
  model_nh_top = 0x40000;
  saves = restores = reenter = 0;
  bosc_ame_private_ddr_saved_mie = 0;
}
static void model_irq_restore(uintptr_t saved) {
  assert(!bosc_ame_private_ddr_busy && !(model_status & 8u));
  assert(!(saved & ~(uintptr_t)8u));
  model_status |= saved;
  ++restores;
}
static void model_shared(void) {
  assert(bosc_ame_private_ddr_busy && !(model_status & 8u));
  if (reenter) {
    reenter = 0;
    bosc_ame_private_ddr_acquire(64);
    assert(!"reentrant entry must fail");
  }
}

static void rejects_context(unsigned kind) {
  reset();
  switch (kind) {
  case 0:
    model_sp = model_nh_bottom + 64;
    break;
  case 1:
    model_e6_bottom = 0;
    break;
  case 2:
    model_e6_top = model_e6_bottom;
    break;
  case 3:
    model_e6_top = model_e6_bottom - 16;
    break;
  case 4:
    ++model_e6_bottom;
    break;
  case 5:
    --model_e6_top;
    break;
  case 6:
    model_e6_top = model_e6_bottom + 16;
    break;
  case 7:
    model_nh_bottom = model_e6_bottom;
    break;
  case 8:
    model_nh_top = 0;
    break;
  case 9:
    model_sp = model_e6_top;
    break;
  case 10:
    model_sp = model_e6_bottom - 1;
    break;
  default:
    assert(0);
  }
  FAILS(BOSC_AME_PRIVATE_DDR_FAIL_OWNER, bosc_ame_private_ddr_acquire(64));
  assert(!saves && !restores);
  assert(model_status == (UINT64_C(0x0601e600) | incoming_mie));
  assert(!bosc_ame_private_ddr_busy && !bosc_ame_private_ddr_owner &&
         !bosc_ame_private_ddr_top && !bosc_ame_private_ddr_active &&
         !bosc_ame_private_ddr_acquires);
  for (unsigned i = 0; i != sizeof(arena); ++i)
    assert(arena[i] == 0xa5);
}

int main(void) {
  for (volatile unsigned mode = 0; mode != 2; ++mode) {
    incoming_mie = mode ? 8u : 0u;
    generic_frame_cases();
    reset();
    uint64_t outer = bosc_ame_private_ddr_acquire(128);
    assert(model_status == (UINT64_C(0x0601e600) | incoming_mie));
    uint64_t inner = bosc_ame_private_ddr_acquire(64);
    assert(model_status == (UINT64_C(0x0601e600) | incoming_mie));
    bosc_ame_private_ddr_release(inner, 64);
    bosc_ame_private_ddr_release(outer, 128);
    assert(saves == 4 && restores == 4 && !bosc_ame_private_ddr_busy);
    assert(model_status == (UINT64_C(0x0601e600) | incoming_mie));
    assert(!bosc_ame_private_ddr_owner && !bosc_ame_private_ddr_top);
    for (unsigned kind = 0; kind != 11; ++kind)
      rejects_context(kind);
    reset();
    hart = 1;
    FAILS(BOSC_AME_PRIVATE_DDR_FAIL_OWNER, bosc_ame_private_ddr_acquire(64));
    assert(!saves && !restores && !bosc_ame_private_ddr_busy);
    reset();
    bosc_ame_private_ddr_saved_mie = 0xdead;
    bosc_ame_private_ddr_busy = 1;
    FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CONCURRENT,
          bosc_ame_private_ddr_acquire(64));
    assert(bosc_ame_private_ddr_saved_mie == 0xdead);
    assert(saves == 1 && !restores);
    reset();
    outer = bosc_ame_private_ddr_acquire(128);
    uintptr_t saved_top = bosc_ame_private_ddr_top;
    unsigned char saved_frame[256];
    memcpy(saved_frame, arena, sizeof(saved_frame));
    bosc_ame_private_ddr_owner = 2;
    FAILS(BOSC_AME_PRIVATE_DDR_FAIL_OWNER, bosc_ame_private_ddr_acquire(64));
    assert(bosc_ame_private_ddr_active == outer &&
           bosc_ame_private_ddr_top == saved_top);
    assert(bosc_ame_private_ddr_acquires == 1 &&
           !bosc_ame_private_ddr_releases && bosc_ame_private_ddr_owner == 2);
    assert(!memcmp(saved_frame, arena, sizeof(saved_frame)));
    reset();
    reenter = 1;
    FAILS(BOSC_AME_PRIVATE_DDR_FAIL_CONCURRENT,
          bosc_ame_private_ddr_acquire(64));
    assert(bosc_ame_private_ddr_saved_mie == incoming_mie);
    assert(saves == 2 && !restores && bosc_ame_private_ddr_busy);
    assert(!bosc_ame_private_ddr_top && !bosc_ame_private_ddr_active);
    reset();
    outer = bosc_ame_private_ddr_acquire(128);
    model_status |= UINT64_C(0x100000000);
    bosc_ame_private_ddr_release(outer, 128);
    assert(model_status == (UINT64_C(0x10601e600) | incoming_mie));
  }
  puts("E6 private DDR: context, MIE, reentry and nested frame checks passed");
  return 0;
}
