//===- bare_runtime.h - Bare-metal C runtime interface --------------------===//
//
// Banner / main hooks used by hello.c. Adapted from ModelZoo
// examples/tools/bare_runtime.h.
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_BARE_RUNTIME_H
#define EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_BARE_RUNTIME_H

void bare_runtime_print_banner(void);
void bare_runtime_before_main(void);
void bare_runtime_after_main(void);

#endif // EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_BARE_RUNTIME_H
