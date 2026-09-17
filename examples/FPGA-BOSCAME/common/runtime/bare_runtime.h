#ifndef EXAMPLES_TOOLS_BARE_RUNTIME_H
#define EXAMPLES_TOOLS_BARE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

void *bare_runtime_heap_base(void);
size_t bare_runtime_heap_offset(void);
void bare_runtime_preserve_heap(void);
void bare_runtime_reset_heap(void);

void bare_runtime_print_banner(void);
void bare_runtime_before_main(void);
void bare_runtime_after_main(void);

/* Buddy cycle-trace support for bare-metal targets. */
void bare_trace_reset(void);
void bare_trace_print(void);

void ame_fence(void);

#endif
