#ifndef QWEN_UNICODE_TABLES_H
#define QWEN_UNICODE_TABLES_H
#include <stdint.h>

typedef struct { uint32_t first, last; } QwenRange;
typedef struct { uint32_t code, count, sequence[4]; } QwenDecomposition;
typedef struct { uint32_t code, combining; } QwenCombining;
typedef struct { uint32_t first, second, composed; } QwenComposition;

extern const QwenRange qwen_letter_ranges[];
extern const uint32_t qwen_letter_count;
extern const QwenRange qwen_number_ranges[];
extern const uint32_t qwen_number_count;
extern const QwenRange qwen_space_ranges[];
extern const uint32_t qwen_space_count;
extern const QwenDecomposition qwen_canonical[];
extern const uint32_t qwen_canonical_count;
extern const QwenCombining qwen_combining[];
extern const uint32_t qwen_combining_count;
extern const QwenComposition qwen_composition[];
extern const uint32_t qwen_composition_count;

#endif
