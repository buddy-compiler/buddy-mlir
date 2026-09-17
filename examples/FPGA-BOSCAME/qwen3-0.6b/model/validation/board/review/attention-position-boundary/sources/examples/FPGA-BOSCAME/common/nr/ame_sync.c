/* Adapted from ModelZoo examples/tools/bare_runtime.c, commit
 * 8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3. NR uses non-transposed B
 * loads and integer accumulator stores. Every AME instruction is fenced,
 * as in ModelZoo's NR assembly restriction stage. */
#include "nr_runtime.h"
#define AME_MTYPE_INT8  ((1ULL << 16) | (1ULL << 4) | 0x0ULL)
#define AME_MTYPE_INT32 ((1ULL << 16) | (1ULL << 6) | 0x2ULL)
static float ame_sync_cell[1] __attribute__((aligned(64))) = {0.0f};
static int8_t ame_sync_a[1] __attribute__((aligned(64))) = {1};
static int8_t ame_sync_b[1] __attribute__((aligned(64))) = {1};

static inline int ame_msettilem(int rem)
{
    register size_t in asm("a0") = (size_t)rem;
    register size_t out asm("a6");
    __asm__ volatile ("fence rw, rw\n\t.word 0x04055877\n\tfence rw, rw" : "+r"(in), "=r"(out) :: "memory");
    return (int)out;
}

static inline int ame_msettilen(int rem)
{
    register size_t in asm("a0") = (size_t)rem;
    register size_t out asm("t2");
    __asm__ volatile ("fence rw, rw\n\t.word 0x040543f7\n\tfence rw, rw" : "+r"(in), "=r"(out) :: "memory");
    return (int)out;
}

static inline int ame_msettilek(int rem)
{
    register size_t inout asm("a3") = (size_t)rem;
    __asm__ volatile ("fence rw, rw\n\t.word 0x0406e6f7\n\tfence rw, rw" : "+r"(inout) :: "memory");
    return (int)inout;
}

static inline void ame_msettype(uint64_t mtype)
{
    register uint64_t a0 asm("a0") = mtype;
    __asm__ volatile ("fence rw, rw\n\t.word 0x00054077\n\tfence rw, rw" :: "r"(a0) : "memory");
    __asm__ volatile ("" ::: "a0", "a1", "a2", "a3", "a4", "a5", "a6",
                      "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
                      "memory");
}

static inline void ame_mlae8(const int8_t *base, int stride_bytes)
{
    register const int8_t *a0 asm("a0") = base;
    register size_t a1 asm("a1") = (size_t)stride_bytes;
    __asm__ volatile ("fence rw, rw\n\t.word 0x04b50077\n\tfence rw, rw" :: "r"(a0), "r"(a1) : "memory");
}

static inline void ame_mlbt8(const int8_t *base, int stride_bytes)
{
    register const int8_t *a0 asm("a0") = base;
    register size_t a1 asm("a1") = (size_t)stride_bytes;
    /* The resync tile is exactly 1x1: ordinary and transposed B loads have
     * identical layout. NR FPGA0 traps on the transposed form. */
    __asm__ volatile ("fence rw, rw\n\t.word 0x08b500f7\n\tfence rw, rw" :: "r"(a0), "r"(a1) : "memory");
}

static inline void ame_mqma_b(void)
{
    __asm__ volatile ("fence rw, rw\n\t.word 0x28180877\n\tfence rw, rw" ::: "memory");
}

static inline void ame_mlce32(const float *base, int stride_bytes)
{
    register const float *t3 asm("t3") = base;
    register size_t t0 asm("t0") = (size_t)stride_bytes;
    __asm__ volatile ("fence rw, rw\n\t.word 0x005e2077\n\tfence rw, rw" :: "r"(t3), "r"(t0) : "memory");
}

static inline void ame_msce32(float *base, int stride_bytes)
{
    register float *t3 asm("t3") = base;
    register size_t t0 asm("t0") = (size_t)stride_bytes;
    __asm__ volatile ("fence rw, rw\n\t.word 0x025e2077\n\tfence rw, rw" :: "r"(t3), "r"(t0) : "memory");
}

static void ame_resync_state(void)
{
    (void)ame_msettilem(1);
    (void)ame_msettilen(1);
    (void)ame_msettilek(1);

    ame_msettype(AME_MTYPE_INT8);
    ame_mlae8(ame_sync_a, (int)sizeof(int8_t));
    ame_mlbt8(ame_sync_b, (int)sizeof(int8_t));
    ame_mqma_b();

    ame_msettype(AME_MTYPE_INT32);
    ame_mlce32(ame_sync_cell, (int)sizeof(float));
    ame_msce32(ame_sync_cell, (int)sizeof(float));
}

void ame_fence(void)
{
    __asm__ volatile ("fence rw, rw" ::: "memory");
    ame_resync_state();
    __asm__ volatile ("fence rw, rw" ::: "memory");
}
