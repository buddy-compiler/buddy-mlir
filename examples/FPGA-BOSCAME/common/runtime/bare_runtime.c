#include "bare_runtime.h"
#include "uart.h"
#include <stddef.h>
#include <stdint.h>

#ifndef UART_FREQ_HZ
#define UART_FREQ_HZ 50000000u
#endif

#ifndef UART_BAUD
#define UART_BAUD 115200u
#endif

#ifndef BARE_RUNTIME_HEAP_SIZE
#define BARE_RUNTIME_HEAP_SIZE (128u * 1024u)
#endif

#ifndef BARE_RUNTIME_HEAP_ALIGN
#define BARE_RUNTIME_HEAP_ALIGN 64u
#endif

#ifndef BARE_RUNTIME_RECLAIM_FREE
#define BARE_RUNTIME_RECLAIM_FREE 0
#endif

#ifndef BARE_RUNTIME_TRACE_MALLOC
#define BARE_RUNTIME_TRACE_MALLOC 0
#endif

#ifndef BARE_RUNTIME_TRACE_MEMCPY
#define BARE_RUNTIME_TRACE_MEMCPY 0
#endif

#ifndef BARE_RUNTIME_TRACE_MEMREF_COPY
#define BARE_RUNTIME_TRACE_MEMREF_COPY 0
#endif

#ifndef BARE_RUNTIME_ENABLE_MEMREF_COPY
#define BARE_RUNTIME_ENABLE_MEMREF_COPY 1
#endif

#ifndef BARE_RUNTIME_ENABLE_AME
#define BARE_RUNTIME_ENABLE_AME 1
#endif

#ifndef BARE_RUNTIME_TRACE_MAX_ID
#define BARE_RUNTIME_TRACE_MAX_ID 256
#endif

#ifndef BARE_RUNTIME_TRACE_LIVE
#define BARE_RUNTIME_TRACE_LIVE 0
#endif

#ifndef BARE_RUNTIME_MEMREF_COPY_MAX_ELEMS
#define BARE_RUNTIME_MEMREF_COPY_MAX_ELEMS (16u * 1024u * 1024u)
#endif

#ifndef BARE_RUNTIME_MEMREF_COPY_MAX_RANK
#define BARE_RUNTIME_MEMREF_COPY_MAX_RANK 8
#endif

#if BARE_RUNTIME_ENABLE_AME
#define AME_MTYPE_INT8  ((1ULL << 16) | (1ULL << 4) | 0x0ULL)
#define AME_MTYPE_INT32 ((1ULL << 16) | (1ULL << 6) | 0x2ULL)
#endif

#if BARE_RUNTIME_TRACE_MALLOC
#define TRACE_MALLOC(msg) print_uart(msg)
#else
#define TRACE_MALLOC(msg) ((void)0)
#endif

#if BARE_RUNTIME_TRACE_MEMCPY
#define TRACE_MEMCPY(msg) print_uart(msg)
#else
#define TRACE_MEMCPY(msg) ((void)0)
#endif

#if BARE_RUNTIME_TRACE_MEMREF_COPY
#define TRACE_MEMREF_COPY(msg) print_uart(msg)
#else
#define TRACE_MEMREF_COPY(msg) ((void)0)
#endif

#if BARE_RUNTIME_TRACE_MALLOC || BARE_RUNTIME_TRACE_MEMCPY || \
    BARE_RUNTIME_TRACE_MEMREF_COPY
static void trace_hex_u64(uint64_t value)
{
    print_uart("0x");
    print_uart_addr(value);
}
#endif

static uint8_t heap_buf[BARE_RUNTIME_HEAP_SIZE]
    __attribute__((aligned(BARE_RUNTIME_HEAP_ALIGN),
                   section(".bare_runtime_heap")));
static size_t heap_offset = 0;
static size_t heap_floor = 0;
#if BARE_RUNTIME_RECLAIM_FREE
typedef struct {
    size_t size;
    size_t is_free;
} BareHeapBlock;
#define BARE_HEAP_HEADER_SIZE                                                \
    ((sizeof(BareHeapBlock) + BARE_RUNTIME_HEAP_ALIGN - 1u) &               \
     ~(size_t)(BARE_RUNTIME_HEAP_ALIGN - 1u))
#endif
typedef struct {
    uint64_t start;
    uint64_t total;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t count;
    int64_t depth;
    int64_t path[4];
} BareTraceRecord;
static BareTraceRecord trace_records[BARE_RUNTIME_TRACE_MAX_ID];
static uint64_t trace_invalid_ids = 0;
#if BARE_RUNTIME_ENABLE_AME
static float ame_sync_cell[1] __attribute__((aligned(64))) = {0.0f};
static int8_t ame_sync_a[1] __attribute__((aligned(64))) = {1};
static int8_t ame_sync_b[1] __attribute__((aligned(64))) = {1};
#endif

void *bare_runtime_heap_base(void)
{
    return heap_buf;
}

size_t bare_runtime_heap_offset(void)
{
    return heap_offset;
}

void bare_runtime_preserve_heap(void)
{
    heap_floor = heap_offset;
}

void bare_runtime_reset_heap(void)
{
    heap_offset = heap_floor;
}

__attribute__((weak)) void bare_runtime_print_banner(void)
{
}

__attribute__((weak)) void bare_runtime_before_main(void)
{
}

__attribute__((weak)) void bare_runtime_after_main(void)
{
}

static inline uint64_t bare_trace_read_cycle(void)
{
#if defined(__riscv)
    uint64_t value;
    __asm__ volatile("rdcycle %0" : "=r"(value));
    return value;
#else
    return 0;
#endif
}

/*
 * Fixed-width RVV dequantize/accumulate leaf used by the optimized Qwen3
 * W8A8 lowering.  Keep this sequence in inline assembly: on the current FPGA
 * the compiler-generated equivalent can spill vectors through an unsupported
 * vlenb CSR read.  This is the same e32,m1 sequence used by the validated
 * hand-written Qwen3 kernel. Fence producer/consumer boundaries and vector
 * memory accesses. A build without V uses the scalar branch below.
 */
__attribute__((noinline)) void _mlir_ciface_buddy_w8a8_rvv_accumulate_n64(
    uint64_t xout_address, uint64_t crow_address, uint64_t ws_row_address,
    uint32_t xscale_bits)
{
#if defined(__riscv)
    __asm__ volatile("fence rw, rw" ::: "memory");
#endif
#if BARE_RUNTIME_TRACE_LIVE
    static int diagnosed = 0;
    if (!diagnosed) {
        volatile float probe;
        diagnosed = 1;
        print_uart("[w8a8-rvv] xout=0x");
        print_uart_addr(xout_address);
        print_uart(" crow=0x");
        print_uart_addr(crow_address);
        print_uart(" ws=0x");
        print_uart_addr(ws_row_address);
        print_uart("\r\n");
        probe = *(const float *)(uintptr_t)xout_address;
        (void)probe;
        print_uart("[w8a8-rvv] scalar xout ok\r\n");
        probe = *(const float *)(uintptr_t)crow_address;
        (void)probe;
        print_uart("[w8a8-rvv] scalar crow ok\r\n");
        print_uart("[w8a8-rvv] crow lanes=0x");
        print_uart_addr((uint32_t)((const int32_t *)(uintptr_t)crow_address)[0]);
        print_uart(" 0x");
        print_uart_addr((uint32_t)((const int32_t *)(uintptr_t)crow_address)[16]);
        print_uart(" 0x");
        print_uart_addr((uint32_t)((const int32_t *)(uintptr_t)crow_address)[32]);
        print_uart(" 0x");
        print_uart_addr((uint32_t)((const int32_t *)(uintptr_t)crow_address)[48]);
        print_uart("\r\n");
        probe = *(const float *)(uintptr_t)ws_row_address;
        (void)probe;
        print_uart("[w8a8-rvv] scalar ws ok\r\n");
    }
#endif
#if defined(__riscv_vector) && !defined(BARE_RUNTIME_AME_STORE_I32)
    float *xout = (float *)(uintptr_t)xout_address;
    const float *crow = (const float *)(uintptr_t)crow_address;
    const float *ws_row = (const float *)(uintptr_t)ws_row_address;
    int done = 0;

    /* The FPGA needs one complete vector issue group after an AME store before
       floating-point RVV accumulation becomes reliable.  Prime every opcode
       with a zero scale so the sequence writes xout back unchanged. */
    __asm__ volatile(
        "vsetivli zero, 16, e32, m1, ta, ma\n\t"
        "vle32.v v10, (%[ws])\n\t"
        "fence rw, rw\n\t"
        "vle32.v v9, (%[acc])\n\t"
        "fence rw, rw\n\t"
        "vle32.v v8, (%[xout])\n\t"
        "fence rw, rw\n\t"
        "vmv.v.i v11, 0\n\t"
        "vfmul.vv v12, v9, v11\n\t"
        "vfmul.vv v13, v12, v10\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "fence rw, rw\n\t"
        "vse32.v v8, (%[xout])\n\t"
        "fence rw, rw"
        :: [xout] "r"(xout), [acc] "r"(crow), [ws] "r"(ws_row)
        : "v8", "v9", "v10", "v11", "v12", "v13", "memory");

    while (done < 64) {
        size_t vl;
        float *xout_ptr = xout + done;
        const float *acc_ptr = crow + done;
        const float *ws_ptr = ws_row + done;
        const size_t avl = (size_t)(64 - done);

        __asm__ volatile(
            "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
            "fence rw, rw\n\t"
            "vle32.v v10, (%[ws])\n\t"
            "fence rw, rw\n\t"
            "vle32.v v9, (%[acc])\n\t"
            "fence rw, rw\n\t"
            "vle32.v v8, (%[xout])\n\t"
            "fence rw, rw\n\t"
            "vmv.v.x v11, %[xs_bits]\n\t"
            "vfmul.vv v12, v9, v11\n\t"
            "vfmul.vv v13, v12, v10\n\t"
            "vfadd.vv v8, v8, v13\n\t"
            "fence rw, rw\n\t"
            "vse32.v v8, (%[xout])\n\t"
            "fence rw, rw\n\t"
            : [vl] "=&r"(vl)
            : [avl] "r"(avl), [xout] "r"(xout_ptr),
              [acc] "r"(acc_ptr), [ws] "r"(ws_ptr),
              [xs_bits] "r"((unsigned long)xscale_bits)
            : "v8", "v9", "v10", "v11", "v12", "v13", "memory");

        if (vl == 0)
            break;
        done += (int)vl;
    }
#else
    float *xout = (float *)(uintptr_t)xout_address;
#if defined(BARE_RUNTIME_AME_STORE_I32)
    /* NR FPGA0 msce32 stores signed integer accumulator bits. Convert their
     * numeric value before applying the model's unchanged FP32 scales. */
    typedef int32_t AmeI32Word __attribute__((__may_alias__));
    const AmeI32Word *crow = (const AmeI32Word *)(uintptr_t)crow_address;
#else
    const float *crow = (const float *)(uintptr_t)crow_address;
#endif
    const float *ws_row = (const float *)(uintptr_t)ws_row_address;
    union {
        uint32_t u;
        float f;
    } xscale = {xscale_bits};
    for (int i = 0; i < 64; ++i)
        xout[i] += ((float)crow[i] * xscale.f) * ws_row[i];
#endif
#if defined(__riscv)
    __asm__ volatile("fence rw, rw" ::: "memory");
#endif
}

static BareTraceRecord *bare_trace_record(int64_t id)
{
    if (id < 0 || id >= BARE_RUNTIME_TRACE_MAX_ID) {
        ++trace_invalid_ids;
        return (BareTraceRecord *)0;
    }
    return &trace_records[id];
}

void _mlir_ciface_buddyTraceCycleStartPath(
    int64_t id, int64_t depth, int64_t path0, int64_t path1,
    int64_t path2, int64_t path3)
{
    BareTraceRecord *record = bare_trace_record(id);
    if (!record)
        return;
    record->depth = depth;
    record->path[0] = path0;
    record->path[1] = path1;
    record->path[2] = path2;
    record->path[3] = path3;
    record->start = bare_trace_read_cycle();
#if BARE_RUNTIME_TRACE_LIVE
    print_uart("[buddy-trace-live] start id=0x");
    print_uart_addr((uint64_t)id);
    print_uart(" path0=0x");
    print_uart_addr((uint64_t)path0);
    print_uart("\r\n");
#endif
}

void _mlir_ciface_buddyTraceCycleEndPath(
    int64_t id, int64_t depth, int64_t path0, int64_t path1,
    int64_t path2, int64_t path3)
{
    uint64_t end = bare_trace_read_cycle();
    BareTraceRecord *record = bare_trace_record(id);
    uint64_t elapsed;
    (void)depth;
    (void)path0;
    (void)path1;
    (void)path2;
    (void)path3;
    if (!record)
        return;
    elapsed = end - record->start;
    record->total += elapsed;
    if (record->count == 0 || elapsed < record->minimum)
        record->minimum = elapsed;
    if (elapsed > record->maximum)
        record->maximum = elapsed;
    ++record->count;
#if BARE_RUNTIME_TRACE_LIVE
    // Read the counter before UART output so live diagnostics do not become
    // part of the measured region. This mode is intended for locating a slow
    // or stuck region and is disabled in normal performance images.
    print_uart("[buddy-trace-live] id=0x");
    print_uart_addr((uint64_t)id);
    print_uart(" elapsed=0x");
    print_uart_addr(elapsed);
    print_uart(" count=0x");
    print_uart_addr(record->count);
    print_uart("\r\n");
#endif
}

void bare_trace_reset(void)
{
    for (int64_t id = 0; id < BARE_RUNTIME_TRACE_MAX_ID; ++id) {
        trace_records[id].start = 0;
        trace_records[id].total = 0;
        trace_records[id].minimum = 0;
        trace_records[id].maximum = 0;
        trace_records[id].count = 0;
        trace_records[id].depth = 0;
        for (int64_t level = 0; level < 4; ++level)
            trace_records[id].path[level] = -1;
    }
    trace_invalid_ids = 0;
}

void bare_trace_print(void)
{
    int printed_header = 0;
    for (int64_t id = 0; id < BARE_RUNTIME_TRACE_MAX_ID; ++id) {
        BareTraceRecord *record = &trace_records[id];
        if (record->count == 0)
            continue;
        if (!printed_header) {
            print_uart("[buddy-trace] cycle summary\r\n");
            printed_header = 1;
        }
        print_uart("[buddy-trace] id=0x");
        print_uart_addr((uint64_t)id);
        print_uart(" path=");
        for (int64_t level = 0; level < record->depth && level < 4; ++level) {
            if (level)
                write_serial('.');
            print_uart_addr((uint64_t)record->path[level]);
        }
        print_uart(" count=0x");
        print_uart_addr(record->count);
        print_uart(" total=0x");
        print_uart_addr(record->total);
        print_uart(" min=0x");
        print_uart_addr(record->minimum);
        print_uart(" max=0x");
        print_uart_addr(record->maximum);
        print_uart("\r\n");
    }
    if (trace_invalid_ids) {
        print_uart("[buddy-trace] invalid_ids=0x");
        print_uart_addr(trace_invalid_ids);
        print_uart("\r\n");
    }
}

void *malloc(size_t size)
{
#if BARE_RUNTIME_TRACE_MALLOC
    TRACE_MALLOC("rt: malloc size=");
    trace_hex_u64((uint64_t)size);
#endif
    size = (size + (BARE_RUNTIME_HEAP_ALIGN - 1u)) &
           ~(size_t)(BARE_RUNTIME_HEAP_ALIGN - 1u);
#if BARE_RUNTIME_RECLAIM_FREE
    size_t block_offset = heap_floor;
    while (block_offset < heap_offset) {
        BareHeapBlock *block = (BareHeapBlock *)&heap_buf[block_offset];
        if (block->is_free && block->size >= size) {
            size_t remainder = block->size - size;
            if (remainder >= BARE_HEAP_HEADER_SIZE + BARE_RUNTIME_HEAP_ALIGN) {
                size_t next_offset =
                    block_offset + BARE_HEAP_HEADER_SIZE + size;
                BareHeapBlock *next =
                    (BareHeapBlock *)&heap_buf[next_offset];
                next->size = remainder - BARE_HEAP_HEADER_SIZE;
                next->is_free = 1;
                block->size = size;
            }
            block->is_free = 0;
            return &heap_buf[block_offset + BARE_HEAP_HEADER_SIZE];
        }
        block_offset += BARE_HEAP_HEADER_SIZE + block->size;
    }
    if (heap_offset + BARE_HEAP_HEADER_SIZE + size > sizeof(heap_buf))
        return (void *)0;
    BareHeapBlock *block = (BareHeapBlock *)&heap_buf[heap_offset];
    block->size = size;
    block->is_free = 0;
    void *ptr = &heap_buf[heap_offset + BARE_HEAP_HEADER_SIZE];
    heap_offset += BARE_HEAP_HEADER_SIZE + size;
    return ptr;
#else
    if (heap_offset + size > sizeof(heap_buf)) {
#if BARE_RUNTIME_TRACE_MALLOC
        TRACE_MALLOC(" failed off=");
        trace_hex_u64((uint64_t)heap_offset);
        TRACE_MALLOC("\r\n");
#endif
        return (void *)0;
    }

    void *ptr = &heap_buf[heap_offset];
    heap_offset += size;
#if BARE_RUNTIME_TRACE_MALLOC
    TRACE_MALLOC(" ptr=");
    trace_hex_u64((uint64_t)(uintptr_t)ptr);
    TRACE_MALLOC(" next=");
    trace_hex_u64((uint64_t)heap_offset);
    TRACE_MALLOC("\r\n");
#endif
    return ptr;
#endif
}

void free(void *ptr)
{
#if BARE_RUNTIME_RECLAIM_FREE
    if (!ptr || (uint8_t *)ptr < heap_buf + heap_floor + BARE_HEAP_HEADER_SIZE ||
        (uint8_t *)ptr >= heap_buf + heap_offset)
        return;
    size_t target = (size_t)((uint8_t *)ptr - heap_buf) - BARE_HEAP_HEADER_SIZE;
    size_t current = heap_floor;
    size_t previous = (size_t)-1;
    while (current < heap_offset && current != target) {
        BareHeapBlock *block = (BareHeapBlock *)&heap_buf[current];
        previous = current;
        current += BARE_HEAP_HEADER_SIZE + block->size;
    }
    if (current != target)
        return;
    BareHeapBlock *block = (BareHeapBlock *)&heap_buf[current];
    block->is_free = 1;
    size_t next_offset = current + BARE_HEAP_HEADER_SIZE + block->size;
    if (next_offset < heap_offset) {
        BareHeapBlock *next = (BareHeapBlock *)&heap_buf[next_offset];
        if (next->is_free)
            block->size += BARE_HEAP_HEADER_SIZE + next->size;
    }
    if (previous != (size_t)-1) {
        BareHeapBlock *prev = (BareHeapBlock *)&heap_buf[previous];
        if (prev->is_free) {
            prev->size += BARE_HEAP_HEADER_SIZE + block->size;
            current = previous;
            block = prev;
        }
    }
    if (current + BARE_HEAP_HEADER_SIZE + block->size == heap_offset)
        heap_offset = current;
#else
    (void)ptr;
#endif
}

static void byte_copy(char *dst, const char *src, int64_t size)
{
    for (int64_t i = 0; i < size; ++i)
        dst[i] = src[i];
}

void *memcpy(void *dst, const void *src, size_t size)
{
#if BARE_RUNTIME_TRACE_MEMCPY
    TRACE_MEMCPY("rt: memcpy size=");
    trace_hex_u64((uint64_t)size);
    TRACE_MEMCPY(" dst=");
    trace_hex_u64((uint64_t)(uintptr_t)dst);
    TRACE_MEMCPY(" src=");
    trace_hex_u64((uint64_t)(uintptr_t)src);
    TRACE_MEMCPY("\r\n");
#endif
    byte_copy((char *)dst, (const char *)src, (int64_t)size);
    return dst;
}

void *memmove(void *dst, const void *src, size_t size)
{
    char *d = (char *)dst;
    const char *s = (const char *)src;
    if (d <= s) {
        byte_copy(d, s, (int64_t)size);
    } else {
        for (size_t i = size; i > 0; --i)
            d[i - 1] = s[i - 1];
    }
    return dst;
}

void *memset(void *dst, int value, size_t size)
{
    char *d = (char *)dst;
    for (size_t i = 0; i < size; ++i)
        d[i] = (char)value;
    return dst;
}

size_t strlen(const char *text)
{
    const char *end = text;
    while (*end)
        ++end;
    return (size_t)(end - text);
}

int strcmp(const char *lhs, const char *rhs)
{
    while (*lhs && *lhs == *rhs) {
        ++lhs;
        ++rhs;
    }
    return (int)(unsigned char)*lhs - (int)(unsigned char)*rhs;
}

typedef struct {
    int64_t rank;
    void *descriptor;
} UnrankedMemRef;

typedef struct {
    char *base;
    char *data;
    int64_t offset;
    int64_t sizes_and_strides[];
} RankedMemRef;

#if BARE_RUNTIME_TRACE_MEMREF_COPY
static void trace_memref_i64(const char *name, int64_t value)
{
    TRACE_MEMREF_COPY(name);
    trace_hex_u64((uint64_t)value);
}

static void trace_memref_ptr(const char *name, const void *ptr)
{
    TRACE_MEMREF_COPY(name);
    trace_hex_u64((uint64_t)(uintptr_t)ptr);
}

static void trace_ranked_memref(const char *label, const RankedMemRef *memref,
                                int64_t rank)
{
    TRACE_MEMREF_COPY(label);
    trace_memref_ptr(" base=", memref->base);
    trace_memref_ptr(" data=", memref->data);
    trace_memref_i64(" off=", memref->offset);
    for (int64_t dim = 0; dim < rank; ++dim) {
        trace_memref_i64(" size=", memref->sizes_and_strides[dim]);
        trace_memref_i64(
            " stride=", memref->sizes_and_strides[rank + dim]);
    }
    TRACE_MEMREF_COPY("\r\n");
}
#endif

#if BARE_RUNTIME_ENABLE_MEMREF_COPY
/* The aligned case covers the FP32 buffers produced by MLIR.  may_alias keeps
 * these bitwise word copies valid for float, integer, and byte allocations. */
typedef uint32_t BareMemRefCopyWord __attribute__((__may_alias__));

static void contiguous_memref_copy(char *dst, const char *src, int64_t bytes)
{
    if ((((uintptr_t)dst | (uintptr_t)src) & 3u) == 0) {
        BareMemRefCopyWord *out = (BareMemRefCopyWord *)dst;
        const BareMemRefCopyWord *in = (const BareMemRefCopyWord *)src;
        int64_t words = bytes / 4;
        for (int64_t i = 0; i < words; ++i)
            out[i] = in[i];
        dst += words * 4;
        src += words * 4;
        bytes -= words * 4;
    }
    byte_copy(dst, src, bytes);
}

void memrefCopy(int64_t elemSize, UnrankedMemRef *srcArg,
                UnrankedMemRef *dstArg)
{
    TRACE_MEMREF_COPY("rt: copy begin elem=");
#if BARE_RUNTIME_TRACE_MEMREF_COPY
    trace_hex_u64((uint64_t)elemSize);
    TRACE_MEMREF_COPY("\r\n");
#endif

    if (!srcArg || !dstArg || srcArg->rank != dstArg->rank || elemSize <= 0) {
        TRACE_MEMREF_COPY("rt: copy skipped\r\n");
        return;
    }

    int64_t rank = srcArg->rank;
    if (rank < 0 || rank > BARE_RUNTIME_MEMREF_COPY_MAX_RANK) {
        TRACE_MEMREF_COPY("rt: copy skipped rank\r\n");
        return;
    }

#if BARE_RUNTIME_TRACE_MEMREF_COPY
    trace_memref_i64("rt: copy rank=", rank);
    TRACE_MEMREF_COPY("\r\n");
#endif

    RankedMemRef *src = (RankedMemRef *)srcArg->descriptor;
    RankedMemRef *dst = (RankedMemRef *)dstArg->descriptor;
    if (!src || !dst || !src->data || !dst->data) {
        TRACE_MEMREF_COPY("rt: copy skipped\r\n");
        return;
    }

#if BARE_RUNTIME_TRACE_MEMREF_COPY
    trace_ranked_memref("rt: copy src", src, rank);
    trace_ranked_memref("rt: copy dst", dst, rank);
#endif

    int64_t *src_sizes = src->sizes_and_strides;
    int64_t *src_strides = src_sizes + rank;
    int64_t *dst_sizes = dst->sizes_and_strides;
    int64_t *dst_strides = dst_sizes + rank;
    uint64_t elements = 1;
    for (int64_t dim = 0; dim < rank; ++dim) {
        if (src_sizes[dim] != dst_sizes[dim] || src_sizes[dim] < 0) {
            TRACE_MEMREF_COPY("rt: copy skipped shape\r\n");
            return;
        }
        if (src_sizes[dim] == 0) {
            TRACE_MEMREF_COPY("rt: copy end empty\r\n");
            return;
        }
        if (elements > (uint64_t)BARE_RUNTIME_MEMREF_COPY_MAX_ELEMS /
                           (uint64_t)src_sizes[dim]) {
            TRACE_MEMREF_COPY("rt: copy skipped too large\r\n");
            return;
        }
        elements *= (uint64_t)src_sizes[dim];
    }

    if (elements > (uint64_t)INT64_MAX / (uint64_t)elemSize) {
        TRACE_MEMREF_COPY("rt: copy skipped byte size\r\n");
        return;
    }

    /* Singleton dimensions address only coordinate zero, so their strides do
     * not affect contiguity.  Negative strides on non-singleton dimensions
     * retain the generic path below, as do transposes and slices with gaps. */
    int contiguous = 1;
    int64_t expected_stride = 1;
    for (int64_t dim = rank; dim > 0; --dim) {
        int64_t current = dim - 1;
        if (src_sizes[current] > 1 &&
            (src_strides[current] != expected_stride ||
             dst_strides[current] != expected_stride)) {
            contiguous = 0;
            break;
        }
        expected_stride *= src_sizes[current];
    }
    if (contiguous) {
        contiguous_memref_copy(dst->data + dst->offset * elemSize,
                               src->data + src->offset * elemSize,
                               (int64_t)elements * elemSize);
        TRACE_MEMREF_COPY("rt: copy end contiguous\r\n");
        return;
    }

    for (uint64_t linear = 0; linear < elements; ++linear) {
        uint64_t remaining = linear;
        int64_t src_index = src->offset;
        int64_t dst_index = dst->offset;
        for (int64_t dim = rank; dim > 0; --dim) {
            int64_t current = dim - 1;
            uint64_t coordinate =
                remaining % (uint64_t)src_sizes[current];
            remaining /= (uint64_t)src_sizes[current];
            src_index += (int64_t)coordinate * src_strides[current];
            dst_index += (int64_t)coordinate * dst_strides[current];
        }
        byte_copy(dst->data + dst_index * elemSize,
                  src->data + src_index * elemSize, elemSize);
    }

    TRACE_MEMREF_COPY("rt: copy end\r\n");
}
#endif

#if BARE_RUNTIME_ENABLE_AME
static inline int ame_msettilem(int rem)
{
    register size_t in asm("a0") = (size_t)rem;
    register size_t out asm("a6");
    __asm__ volatile (".word 0x04055877" : "+r"(in), "=r"(out) :: "memory");
    return (int)out;
}

static inline int ame_msettilen(int rem)
{
    register size_t in asm("a0") = (size_t)rem;
    register size_t out asm("t2");
    __asm__ volatile (".word 0x040543f7" : "+r"(in), "=r"(out) :: "memory");
    return (int)out;
}

static inline int ame_msettilek(int rem)
{
    register size_t inout asm("a3") = (size_t)rem;
    __asm__ volatile (".word 0x0406e6f7" : "+r"(inout) :: "memory");
    return (int)inout;
}

static inline void ame_msettype(uint64_t mtype)
{
    register uint64_t a0 asm("a0") = mtype;
    __asm__ volatile (".word 0x00054077" :: "r"(a0) : "memory");
    __asm__ volatile ("" ::: "a0", "a1", "a2", "a3", "a4", "a5", "a6",
                      "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
                      "memory");
}

static inline void ame_mlae8(const int8_t *base, int stride_bytes)
{
    register const int8_t *a0 asm("a0") = base;
    register size_t a1 asm("a1") = (size_t)stride_bytes;
    __asm__ volatile (".word 0x04b50077" :: "r"(a0), "r"(a1) : "memory");
}

static inline void ame_mlbt8(const int8_t *base, int stride_bytes)
{
    register const int8_t *a0 asm("a0") = base;
    register size_t a1 asm("a1") = (size_t)stride_bytes;
#if defined(BARE_RUNTIME_AME_SYNC_NO_TRANSPOSE)
    /* The resync tile is exactly 1x1: ordinary and transposed B loads have
     * identical layout. NR FPGA0 traps on the transposed form. */
    __asm__ volatile (".word 0x08b500f7" :: "r"(a0), "r"(a1) : "memory");
#else
    __asm__ volatile (".word 0x08b508f7" :: "r"(a0), "r"(a1) : "memory");
#endif
}

static inline void ame_mqma_b(void)
{
    __asm__ volatile (".word 0x28180877" ::: "memory");
}

static inline void ame_mlce32(const float *base, int stride_bytes)
{
    register const float *t3 asm("t3") = base;
    register size_t t0 asm("t0") = (size_t)stride_bytes;
    __asm__ volatile (".word 0x005e2077" :: "r"(t3), "r"(t0) : "memory");
}

static inline void ame_msce32(float *base, int stride_bytes)
{
    register float *t3 asm("t3") = base;
    register size_t t0 asm("t0") = (size_t)stride_bytes;
    __asm__ volatile (".word 0x025e2077" :: "r"(t3), "r"(t0) : "memory");
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
#endif

#if !defined(BARE_RUNTIME_HOST_TEST)
__attribute__((weak)) uintptr_t handle_trap(uintptr_t mcause, uintptr_t mepc, uintptr_t regs)
{
    (void)regs;

    print_uart("\r\nTRAP mcause=0x");
    print_uart_addr((uint64_t)mcause);
    print_uart(" mepc=0x");
    print_uart_addr((uint64_t)mepc);
    uintptr_t mtval;
    __asm__ volatile("csrr %0, mtval" : "=r"(mtval));
    print_uart(" mtval=0x");
    print_uart_addr((uint64_t)mtval);
    print_uart("\r\n");

    while (1)
        __asm__ volatile ("wfi");
}

#if !defined(BARE_RUNTIME_CUSTOM_ENTRY)
void _init(void)
{
    bare_runtime_reset_heap();
    init_uart(UART_FREQ_HZ, UART_BAUD);

    bare_runtime_print_banner();

    extern int main(void);
    bare_runtime_before_main();
    main();
    bare_runtime_after_main();

    while (1)
        __asm__ volatile ("wfi");
}
#endif
#endif
