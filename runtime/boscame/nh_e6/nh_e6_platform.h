//===- nh_e6_platform.h - NH/E6 shared platform public interface ----------===//
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

#ifndef NH_E6_PLATFORM_H
#define NH_E6_PLATFORM_H

//
// NH/E6 platform support for the BOSCAME examples.
//
// This is the platform behind the patch6/patch7 NH→E6 contract used by the
// SPM/DMA examples (the delivery's `test/src/cases/nh_e6_dma_spm` variant):
// NH boots from 0x80000000, owns the physical UART and the final verdict, and
// programs/releases E6 through the RA SYSCTRL block.  E6 boots from 0x80010000,
// runs the application's `e6_main()` and publishes its completion through
// `DONE`; NH waits with a bounded poll, prints progress markers from `MAILBOX`
// and reports the generic verdict.  The application adds its own verdict-time
// diagnostics through `nh_e6_app_report()`.
//
// It is deliberately NOT `common/nr`: the NR platform uses 0x80010000 as a DDR
// mailbox, keeps a different completion encoding and link layout, and must not
// change.  The two platforms never share an image.
//
// Fixed hardware contract (validated on the board; do not renumber):
//   0x50000000 + 0x100  E6 program counter written by NH
//   0x50000000 + 0x060  per-core control
//   0x50000000 + 0x050  core release
//   0x50000120          E6 completion code (2 = PASS, 0xee = FAIL, 0xed = TRAP)
//   0x50000140          NH final result (0xff = PASS, 0xee = FAIL)
//   0x50000160          patch6 progress mailbox (NH clears it before release,
//                       E6 publishes markers, the terminal value is relayed
//                       here for the VCS completion protocol)
// The UART is the common/uart driver at 0x310b0000; only NH accesses it.
//
// E6 also installs the delivered reference's seven TOR PMP entries and bare
// `satp` in `nh_e6_platform_init()`.  SPM/DMA addresses come from
// `bosc_ame_dma.h`; the DDR staging window comes from two absolute symbols the
// example linker script must export:
//
//   __nh_e6_ddr_window_start   first byte of PMP entry 5   (0x80100000)
//   __nh_e6_ddr_window_end     one past its last byte      (chosen by link.ld)
//
// Keep CPU stacks in a region whose cache policy is stable across both
// modes. The experimental examples end entry 5 at the E6 stack bottom;
// entry 6 remains write-through in both modes. Internal stack packs are DMA
// sources; results return through explicit CPU vector stores.
//
// They are addresses exported by `link.ld` (`extern char []` + `&` in the
// implementation), so the window can not be shrunk to the staging arrays'
// actual size by accident and the PMP setup can not drift from the layout.
//

#include <stdint.h>

// ------------------------------------------------------------------
// Fixed platform addresses and completion codes
// ------------------------------------------------------------------

#ifndef NH_E6_SYSCTRL_BASE
#define NH_E6_SYSCTRL_BASE ((uintptr_t)UINT32_C(0x50000000))
#endif

#define NH_E6_SYSCTRL(offset) (NH_E6_SYSCTRL_BASE + (uintptr_t)(offset))

#define NH_E6_E6_PC_REG NH_E6_SYSCTRL(UINT32_C(0x100))
#define NH_E6_E6_CONTROL_REG NH_E6_SYSCTRL(UINT32_C(0x060))
#define NH_E6_E6_RELEASE_REG NH_E6_SYSCTRL(UINT32_C(0x050))
#define NH_E6_DONE_REG NH_E6_SYSCTRL(UINT32_C(0x120))
#define NH_E6_RESULT_REG NH_E6_SYSCTRL(UINT32_C(0x140))
#define NH_E6_MAILBOX_REG NH_E6_SYSCTRL(UINT32_C(0x160))

// Released by writing 1; the control register is cleared before release.
#define NH_E6_E6_RELEASE_VALUE UINT32_C(1)

// E6 completion codes, published through NH_E6_DONE_REG.
#define NH_E6_DONE_PASS UINT32_C(2)
#define NH_E6_DONE_FAIL UINT32_C(0xee)
#define NH_E6_DONE_TRAP UINT32_C(0xed)

// NH final result, published through NH_E6_RESULT_REG.
#define NH_E6_RESULT_PASS UINT32_C(0xff)
#define NH_E6_RESULT_FAIL UINT32_C(0xee)

//
// Bounded NH wait.  The board-tested value is kept as the default; a host test
// may predefine a smaller count.
//
#ifndef NH_E6_WAIT_ITERATIONS
#define NH_E6_WAIT_ITERATIONS UINT32_C(100000000)
#endif

// ------------------------------------------------------------------
// Platform services (nh_e6_platform.c)
// ------------------------------------------------------------------

//
// NH main flow, called by the boot module's `_start` after the NH stack and
// `mtvec` are set up.  It initializes the UART, prints the start line, clears
// the completion/result/mailbox registers, programs and releases E6 at
// `e6_start`, polls until a terminal code or the bounded timeout, prints the
// verdict plus the platform trap/DMA diagnostics, calls `nh_e6_app_report()`
// and finally relays the verdict to the VCS mailbox.  It returns with the
// protocol published; the boot module owns the halt.
//
void nh_e6_nh_main(void);

//
// NH trap entry body, called by the boot module's trap stub.  It publishes
// 0xee as the result, prints the NH TRAP verdict and relays 0xee to the
// mailbox.  It returns; the boot module halts.
//
void nh_e6_nh_trap(void);

//
// E6 main flow, called by the boot module's `e6_start` after the E6 stack,
// `mstatus` and `mtvec` are configured and the image BSS is cleared.  It first
// runs `nh_e6_platform_init()`, then calls the application's `e6_main()`,
// publishes PASS/FAIL through the DONE register, waits for NH to publish the
// final result and relays it to the mailbox.  The hardware setup therefore
// precedes every application access, including its CSR 0xbc0 policy switches
// and its first DMA/SPM operation.
//
void nh_e6_e6_main(void);

// ------------------------------------------------------------------
// E6 hardware setup and cache policies (nh_e6_platform.c)
// ------------------------------------------------------------------

//
// One-time E6 address-space setup, already executed by `nh_e6_e6_main()`
// before the application entry; applications do not call it.  It installs the
// delivered reference's seven TOR PMP entries, which is what the CSR 0xbc0
// cache policy below and the DMA/AME physical addresses depend on:
//
//   pmpaddr0/1  SPM window [SPM0, SPM1 + 128 KiB) (`bosc_ame_dma.h`)
//   pmpaddr2/3  DMA window [0x60000000, 0x60020000)
//   pmpaddr4/5  DDR staging window, i.e. PMP entry 5 (link script, see above)
//   pmpaddr6    rest of the address space
//   pmpcfg0     0x000f0f0f0f0f0f0f, TOR, R/W/X for all seven entries
//
// then `fence iorw, iorw`, `csrw satp, zero` and `sfence.vma zero, zero` --
// the example passes physical addresses to DMA and AME in bare mode, and the
// PTE cache must be flushed after the mode switch.
//
void nh_e6_platform_init(void);

//
// CSR 0xbc0 cache policies of the delivered E6. Select the write policy before
// CPU DDR stores and uncached loads before reading DMA-written data. A
// read/modify/write needs both transitions; the read policy left by dma_wait
// must not be relied on for subsequent DDR stores. Both write
// `fence iorw, iorw` then `csrw 0xbc0` and must run after
// `nh_e6_platform_init()`:
//
//   nh_e6_cache_writes()         0x0f0f0f0f0f0f0f0f: write-through stores,
//                                cached loads -- the default at phase
//                                boundaries, before DMA staging writes.
//   nh_e6_cache_uncached_loads() 0x0f0f000f000f000f: uncached loads -- before
// reading data the DMA engines have just written (the scalar oracle).
//
void nh_e6_cache_writes(void);
void nh_e6_cache_uncached_loads(void);

//
// Publish one bounded progress marker while E6 is running: the patch6 mailbox
// first, then a post-mortem copy in `e6_diag_state`.  Applications call this at
// their phase boundaries, followed by `nh_e6_cache_writes()` to restore the
// write-through store policy for the next staging writes.
//
void nh_e6_progress(uint32_t marker);

//
// Record a DMA failure (channel, a bosc_ame_dma_fail_reason, raw STATUS and the
// offending token) and stop E6.  Intended as the `bosc_ame_dma_fail_hook`:
// it never returns, so a failed transfer can not be consumed as data.  The
// trap that follows is captured by the E6 trap entry and reported by NH as
// TRAP with the DMA context line.
//
__attribute__((noreturn)) void nh_e6_publish_dma_failure(uint32_t channel,
                                                         uint32_t reason,
                                                         uint64_t status,
                                                         uint64_t src_token);

//
// Line prefixes for the NH report: `[5.5296MHz][<tag>] ` (lifecycle lines) and
// `[<tag>] ` (diagnostic lines), where <tag> is `nh_e6_app_tag()`.  The
// application uses them for its own report lines so the tag has one source.
//
void nh_e6_print_mhz_prefix(void);
void nh_e6_print_tag_prefix(void);

// ------------------------------------------------------------------
// Symbols provided by the boot module (nh_e6_boot.S)
// ------------------------------------------------------------------

// E6 entry point; the example linker script pins it at 0x80010000.
extern void e6_start(void);

// ------------------------------------------------------------------
// Application hooks, provided by the example's driver C file
// ------------------------------------------------------------------

// E6 algorithm entry.  Returns 0 when the application considers it a PASS.
int e6_main(void);

// Short application tag used in every NH report line, e.g. "ame-64x64x64".
const char *nh_e6_app_tag(void);

//
// Application diagnostics printed by NH after the verdict, on every outcome
// (PASS, FAIL, TRAP and TIMEOUT), so a partial run still reports whatever E6
// measured.  The application starts its own lines with the prefix helpers
// above and reads only its own diagnostic cells.
//
void nh_e6_app_report(void);

// ------------------------------------------------------------------
// Platform diagnostic cells (DDR, cleared by e6_start with the BSS)
// ------------------------------------------------------------------

// Captured by the boot module's E6 trap entry before it publishes 0xed.
extern volatile uint64_t e6_diag_mcause;
extern volatile uint64_t e6_diag_mepc;
extern volatile uint64_t e6_diag_mtval;

// Written by nh_e6_publish_dma_failure before the trap.
extern volatile uint32_t e6_diag_dma_channel;
extern volatile uint32_t e6_diag_dma_reason;
extern volatile uint64_t e6_diag_dma_status;
extern volatile uint64_t e6_diag_dma_src_token;

// Last progress marker published by E6 (post-mortem copy, never read by NH).
extern volatile uint32_t e6_diag_state;

#endif // NH_E6_PLATFORM_H
