// BGE-M3 K3 hybrid GEMM kernel: splits each f32 GEMM between the 8 X100
// application cores (private pthread pool, float-accumulator dot-product
// over a cached transposed copy of B) and the 8 A100 compute cores
// (spine-runtime, double-accumulator dot-product over the same transposed
// B). Measured on K3: GEMV ~3.3 GFLOPS, hybrid dot ~20-24 GFLOPS.
//
// NOTE: a private pthread pool is used instead of OpenMP because the model
// may be invoked from inside an OpenMP region (buddy-server worker
// threads); nested OpenMP either serializes (default) or thrashes
// (OMP_NESTED), while pthreads are unaffected.
#include "spert.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>

namespace {

// --- private parallel-for pool (7 workers + calling thread = 8) ----------
constexpr int kNumWorkers = 7;

struct ParPool {
  pthread_t threads[kNumWorkers] = {};
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t startCv = PTHREAD_COND_INITIALIZER;
  pthread_cond_t doneCv = PTHREAD_COND_INITIALIZER;
  bool jobReady = false;
  uint32_t generation = 0;
  uint32_t finished = 0;
  void (*body)(uint32_t m, void *ctx) = nullptr;
  void *ctx = nullptr;
  std::atomic<uint32_t> nextRow{0};
  uint32_t rows = 0;
};

void *pool_worker(void *arg) {
  ParPool *p = static_cast<ParPool *>(arg);
  for (;;) {
    pthread_mutex_lock(&p->mutex);
    while (!p->jobReady)
      pthread_cond_wait(&p->startCv, &p->mutex);
    const uint32_t gen = p->generation;
    const uint32_t rows = p->rows;
    void (*body)(uint32_t, void *) = p->body;
    void *ctx = p->ctx;
    pthread_mutex_unlock(&p->mutex);
    for (;;) {
      const uint32_t m = p->nextRow.fetch_add(1, std::memory_order_relaxed);
      if (m >= rows)
        break;
      body(m, ctx);
    }
    pthread_mutex_lock(&p->mutex);
    if (++p->finished == (uint32_t)kNumWorkers + 1)
      pthread_cond_broadcast(&p->doneCv);
    while (p->generation == gen && p->jobReady)
      pthread_cond_wait(&p->startCv, &p->mutex);
    pthread_mutex_unlock(&p->mutex);
  }
  return nullptr;
}

ParPool g_pool;
bool g_poolStarted = false;

void pool_init_once() {
  if (g_poolStarted)
    return;
  g_poolStarted = true;
  for (int i = 0; i < kNumWorkers; ++i)
    pthread_create(&g_pool.threads[i], nullptr, pool_worker, &g_pool);
}

// Parallel-for over rows: body(m, ctx) for m in [0, rows). The calling
// thread participates; returns after all rows are processed.
// BGE_M3_A100_NOPOOL=1 disables the worker threads (calling thread only)
// for isolation experiments.
void pool_for(uint32_t rows, void (*body)(uint32_t m, void *ctx), void *ctx) {
  if (getenv("BGE_M3_A100_NOPOOL")) {
    for (uint32_t m = 0; m < rows; ++m)
      body(m, ctx);
    return;
  }
  pool_init_once();
  ParPool &p = g_pool;
  pthread_mutex_lock(&p.mutex);
  p.body = body;
  p.ctx = ctx;
  p.rows = rows;
  p.finished = 0;
  p.nextRow.store(0, std::memory_order_relaxed);
  p.jobReady = true;
  ++p.generation;
  pthread_cond_broadcast(&p.startCv);
  pthread_mutex_unlock(&p.mutex);

  // Calling thread also works.
  for (;;) {
    const uint32_t m = p.nextRow.fetch_add(1, std::memory_order_relaxed);
    if (m >= rows)
      break;
    body(m, ctx);
  }
  pthread_mutex_lock(&p.mutex);
  if (++p.finished == (uint32_t)kNumWorkers + 1)
    pthread_cond_broadcast(&p.doneCv);
  while (p.finished < (uint32_t)kNumWorkers + 1)
    pthread_cond_wait(&p.doneCv, &p.mutex);
  p.jobReady = false;
  pthread_cond_broadcast(&p.startCv);
  pthread_mutex_unlock(&p.mutex);
}

// Monotonic clock in seconds.
inline double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Optional per-call timing stats (BGE_M3_A100_TIMING=1).
struct TimingStats {
  bool enabled = false;
  uint32_t calls = 0;
  double launch_s = 0.0;
  double sync_s = 0.0;
  double x100_s = 0.0;
  double transpose_s = 0.0;
  double hash_s = 0.0;
  double fallback_s = 0.0;
} g_timing;

void timing_report() {
  if (!g_timing.enabled)
    return;
  fprintf(stderr,
          "[a100] TIMING calls=%u launch=%.3fs (%.2fms/call) sync=%.3fs "
          "(%.2fms/call) x100=%.3fs transpose=%.3fs hash=%.3fs "
          "fallback=%.3fs\n",
          g_timing.calls, g_timing.launch_s,
          g_timing.calls ? g_timing.launch_s * 1e3 / g_timing.calls : 0.0,
          g_timing.sync_s,
          g_timing.calls ? g_timing.sync_s * 1e3 / g_timing.calls : 0.0,
          g_timing.x100_s, g_timing.transpose_s, g_timing.hash_s,
          g_timing.fallback_s);
  fflush(stderr);
}

// --- dot-product GEMM over transposed B (Bt is [N][K] row-major) ----------
// X100 form: float accumulators, 4-way unrolled (~18-19 GFLOPS on K3).
void dot_rows_f4(const float *A, const float *Bt, float *C, uint32_t m0,
                 uint32_t m1, uint32_t N, uint32_t K) {
  for (uint32_t m = m0; m < m1; ++m) {
    const float *a_row = A + (size_t)m * K;
    for (uint32_t n = 0; n < N; ++n) {
      const float *bt_row = Bt + (size_t)n * K;
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
      uint32_t k = 0;
      for (; k + 4 <= K; k += 4) {
        acc0 += a_row[k] * bt_row[k];
        acc1 += a_row[k + 1] * bt_row[k + 1];
        acc2 += a_row[k + 2] * bt_row[k + 2];
        acc3 += a_row[k + 3] * bt_row[k + 3];
      }
      for (; k < K; ++k)
        acc0 += a_row[k] * bt_row[k];
      C[(size_t)m * N + n] = (acc0 + acc1) + (acc2 + acc3);
    }
  }
}

// A100 form: double accumulator (~8.4 GFLOPS on the A100 cores; the float4
// form is pathological there).
void dot_rows_dbl(const float *A, const float *Bt, float *C, uint32_t m0,
                  uint32_t m1, uint32_t N, uint32_t K) {
  for (uint32_t m = m0; m < m1; ++m) {
    const float *a_row = A + (size_t)m * K;
    for (uint32_t n = 0; n < N; ++n) {
      const float *bt_row = Bt + (size_t)n * K;
      double acc = 0.0;
      for (uint32_t k = 0; k < K; ++k)
        acc += (double)a_row[k] * (double)bt_row[k];
      C[(size_t)m * N + n] = (float)acc;
    }
  }
}

// --- A100 tile kernel ------------------------------------------------------
struct DotArgs {
  const float *A;
  const float *Bt;
  float *C;
  uint32_t M, N, K;
};

void a100_dot_tile(spert::Context *ctx, DotArgs g) {
  const uint32_t tiles = ctx->grid_dim(0);
  const uint32_t tid = ctx->program_id(0);
  const uint32_t rows = (g.M + tiles - 1) / tiles;
  const uint32_t m0 = tid * rows;
  const uint32_t m1 = (m0 + rows < g.M) ? m0 + rows : g.M;
  dot_rows_dbl(g.A, g.Bt, g.C, m0, m1, g.N, g.K);
}

spert::Stream &get_stream() {
  static spert::Stream stream;
  return stream;
}

// One-shot tools (buddy-cli) run a single inference and exit; initializing
// spert costs ~30s there (first Stream construction brings up the A100
// driver) and cannot be amortized. Skip A100 entirely for them — the X100
// pool alone still beats the baseline. buddy-server keeps the model resident
// and preheats the stream in the background instead.
static bool is_one_shot_process() {
  static bool cached = false;
  static bool result = false;
  if (cached)
    return result;
  cached = true;
  FILE *f = fopen("/proc/self/comm", "r");
  char buf[64] = {0};
  if (f) {
    if (fgets(buf, sizeof(buf), f))
      result = strncmp(buf, "buddy-cli", 9) == 0;
    fclose(f);
  }
  return result;
}

// The FIRST spert::Stream construction on K3 takes ~30s (driver bring-up of
// the A100 cluster; verified with TS entry/exit timestamps around the first
// launch_gemm). Doing it inline blocks the first inference. Instead, kick it
// off on a joinable thread when the model .so is dlopen'd (i.e., during
// weight loading) and let launch_gemm fall back to X100-only until the
// stream reports ready.
std::atomic<bool> g_streamReady{false};
pthread_t g_streamInitThread = 0;
bool g_streamInitStarted = false;

void *stream_init_thread(void *) {
  get_stream();
  g_streamReady.store(true, std::memory_order_release);
  return nullptr;
}

__attribute__((constructor)) static void a100_stream_preinit() {
  if (getenv("BGE_M3_A100_DISABLE") || is_one_shot_process())
    return;
  if (pthread_create(&g_streamInitThread, nullptr, stream_init_thread,
                     nullptr) == 0)
    g_streamInitStarted = true;
}

// Join the init thread at unload so the .so code is never torn down under a
// running thread.
__attribute__((destructor)) static void a100_stream_shutdown() {
  if (g_streamInitStarted && g_streamInitThread) {
    pthread_join(g_streamInitThread, nullptr);
    g_streamInitThread = 0;
    g_streamInitStarted = false;
  }
}

// Returns a ready stream or nullptr when not ready yet.
spert::Stream *ready_stream() {
  if (getenv("BGE_M3_A100_DISABLE") || is_one_shot_process())
    return nullptr;
  if (!g_streamInitStarted) {
    g_streamInitStarted = true;
    if (pthread_create(&g_streamInitThread, nullptr, stream_init_thread,
                       nullptr) != 0)
      return nullptr;
  }
  if (!g_streamReady.load(std::memory_order_acquire))
    return nullptr;
  return &get_stream();
}

// --- transposed-B cache -----------------------------------------------------
// CAUTION: B is per-request activation data in the BGE-M3 graph, and the
// request heap (malloc) reuses the same addresses for the same-size buffers
// across requests. Keying the cache by B POINTER therefore returns STALE
// transposes from earlier requests (buddy-server serves each request in a
// detached thread; repeated identical prompts then produce corrupted,
// request-dependent embeddings). The cache is keyed by a 128-bit CONTENT
// hash of B instead, with LRU eviction.
struct BtEntry {
  uint64_t h0 = 0, h1 = 0;
  uint32_t N = 0, K = 0;
  uint64_t lastUse = 0;
  float *Bt = nullptr;
};

constexpr size_t kMaxBt = 256;
BtEntry g_bt[kMaxBt];
size_t g_btCount = 0;
uint64_t g_clock = 0;

static inline uint64_t mix64(uint64_t h, uint64_t x) {
  h ^= x;
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 27;
  h *= 0x94d049bb133111ebULL;
  h ^= h >> 31;
  return h;
}

// 128-bit content hash of the B buffer (alignment-safe word loads).
void content_hash(const float *B, size_t bytes, uint64_t &h0, uint64_t &h1) {
  const size_t words = bytes / 8;
  size_t i = 0;
  h0 = 0x9e3779b97f4a7c15ULL;
  h1 = 0xc2b2ae3d27d4eb4fULL;
  for (; i + 3 < words; i += 4) {
    uint64_t x0, x1, x2, x3;
    memcpy(&x0, (const uint8_t *)B + i * 8, 8);
    memcpy(&x1, (const uint8_t *)B + (i + 1) * 8, 8);
    memcpy(&x2, (const uint8_t *)B + (i + 2) * 8, 8);
    memcpy(&x3, (const uint8_t *)B + (i + 3) * 8, 8);
    h0 = mix64(h0, x0);
    h1 = mix64(h1, x1);
    h0 = mix64(h0, x2);
    h1 = mix64(h1, x3);
  }
  for (; i < words; ++i) {
    uint64_t x;
    memcpy(&x, (const uint8_t *)B + i * 8, 8);
    h0 = mix64(h0, x);
    h1 = mix64(h1, x);
  }
  uint64_t tail = 0;
  memcpy(&tail, (const uint8_t *)B + words * 8, bytes - words * 8);
  if (bytes - words * 8) {
    h0 = mix64(h0, tail);
    h1 = mix64(h1, tail);
  }
  h0 = mix64(h0, (uint64_t)bytes);
  h1 = mix64(h1, (uint64_t)bytes);
}

// Returns a cached (or newly built) [N][K] row-major transpose of B.
// B must be [K][N] row-major and contiguous. Returns nullptr only if the
// Bt allocation fails (caller falls back to GEMV).
struct TransposeCtx {
  const float *B;
  float *Bt;
  uint32_t N, K;
};

void transpose_row(uint32_t n, void *ctx) {
  TransposeCtx *t = static_cast<TransposeCtx *>(ctx);
  const float *bcol = t->B + (size_t)n;
  float *btrow = t->Bt + (size_t)n * t->K;
  for (uint32_t k = 0; k < t->K; ++k)
    btrow[k] = bcol[(size_t)k * t->N];
}

const float *get_bt(const float *B, uint32_t N, uint32_t K) {
  const size_t bytes = (size_t)N * K * sizeof(float);
  // BGE_M3_A100_NOCACHE=1: never cache across calls — transpose into a
  // reusable scratch buffer every call (isolation experiments).
  if (getenv("BGE_M3_A100_NOCACHE")) {
    static float *scratch = nullptr;
    static size_t scratchBytes = 0;
    if (bytes > scratchBytes) {
      free(scratch);
      scratch = (float *)malloc(bytes);
      scratchBytes = bytes;
    }
    if (!scratch)
      return nullptr;
    const double t0 = g_timing.enabled ? now_s() : 0.0;
    TransposeCtx ctx{B, scratch, N, K};
    pool_for(N, transpose_row, &ctx);
    if (g_timing.enabled)
      g_timing.transpose_s += now_s() - t0;
    return scratch;
  }
  const double th0 = g_timing.enabled ? now_s() : 0.0;
  uint64_t h0, h1;
  content_hash(B, bytes, h0, h1);
  if (g_timing.enabled)
    g_timing.hash_s += now_s() - th0;

  // Content-keyed lookup.
  for (size_t i = 0; i < g_btCount; ++i)
    if (g_bt[i].N == N && g_bt[i].K == K && g_bt[i].h0 == h0 &&
        g_bt[i].h1 == h1) {
      g_bt[i].lastUse = ++g_clock;
      return g_bt[i].Bt;
    }

  // Miss: pick a slot (append, or evict LRU when full).
  size_t slot = g_btCount;
  if (slot >= kMaxBt) {
    slot = 0;
    for (size_t i = 1; i < kMaxBt; ++i)
      if (g_bt[i].lastUse < g_bt[slot].lastUse)
        slot = i;
    if (g_bt[slot].N != N || g_bt[slot].K != K) {
      free(g_bt[slot].Bt);
      g_bt[slot].Bt = nullptr;
    }
  } else {
    ++g_btCount;
  }

  const double t0 = g_timing.enabled ? now_s() : 0.0;
  if (!g_bt[slot].Bt) {
    g_bt[slot].Bt = (float *)malloc(bytes);
    if (!g_bt[slot].Bt)
      return nullptr;
  }
  TransposeCtx ctx{B, g_bt[slot].Bt, N, K};
  pool_for(N, transpose_row, &ctx);
  if (g_timing.enabled)
    g_timing.transpose_s += now_s() - t0;
  g_bt[slot].h0 = h0;
  g_bt[slot].h1 = h1;
  g_bt[slot].N = N;
  g_bt[slot].K = K;
  g_bt[slot].lastUse = ++g_clock;
  return g_bt[slot].Bt;
}

// --- strided GEMV fallback (non-contiguous operands) -----------------------
void gemv_rows(const float *A, const float *B, float *C, uint32_t m0,
               uint32_t m1, uint32_t N, uint32_t K, int64_t a_s0,
               int64_t a_s1, int64_t b_s0, int64_t b_s1, int64_t c_s0,
               int64_t c_s1) {
  constexpr uint32_t VW = 32;
  for (uint32_t m = m0; m < m1; ++m) {
    for (uint32_t n0 = 0; n0 < N; n0 += VW) {
      float acc[VW] = {0};
      for (uint32_t k = 0; k < K; ++k) {
        const float a = A[m * a_s0 + k * a_s1];
        for (uint32_t j = 0; j < VW && n0 + j < N; ++j)
          acc[j] += a * B[k * b_s0 + (n0 + j) * b_s1];
      }
      for (uint32_t j = 0; j < VW && n0 + j < N; ++j)
        C[m * c_s0 + (n0 + j) * c_s1] = acc[j];
    }
  }
}

} // namespace

// Packed 2D memref descriptor (MLIR internal ABI, passed by pointer via the
// C interface wrapper).
struct Memref2DF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

// --- pool body helpers -----------------------------------------------------
struct GemvCtx {
  const float *A;
  const float *B;
  float *C;
  uint32_t N, K;
  int64_t a_s0, a_s1, b_s0, b_s1, c_s0, c_s1;
};

void gemv_one_row(uint32_t m, void *ctx) {
  GemvCtx *g = static_cast<GemvCtx *>(ctx);
  gemv_rows(g->A, g->B, g->C, m, m + 1, g->N, g->K, g->a_s0, g->a_s1, g->b_s0,
            g->b_s1, g->c_s0, g->c_s1);
}

struct DotCtx {
  const float *A;
  const float *Bt;
  float *C;
  uint32_t N, K;
};

void dot_one_row(uint32_t m, void *ctx) {
  DotCtx *d = static_cast<DotCtx *>(ctx);
  dot_rows_f4(d->A, d->Bt, d->C, m, m + 1, d->N, d->K);
}

// Hybrid entry: X100 OpenMP threads compute ~80% of the rows with the float4
// dot form while the A100 cores compute the remaining ~20% with the double
// dot form, both over a cached transposed B.
void launch_gemm(const Memref2DF32 *A, const Memref2DF32 *B,
                 const Memref2DF32 *C) {
  const uint32_t M = (uint32_t)C->sizes[0];
  const uint32_t N = (uint32_t)C->sizes[1];
  const uint32_t K = (uint32_t)A->sizes[1];
  const float *a_ptr = A->aligned + A->offset;
  const float *b_ptr = B->aligned + B->offset;
  float *c_ptr = C->aligned + C->offset;

  spert::Stream *stream = ready_stream();
  const bool useA100 = stream && !getenv("BGE_M3_A100_DISABLE");

  // Fast path requires contiguous row-major operands.
  const bool contiguous =
      A->strides[1] == 1 && B->strides[1] == 1 && C->strides[1] == 1;
  if (!contiguous) {
    const double t0 = g_timing.enabled ? now_s() : 0.0;
    GemvCtx gctx{a_ptr, b_ptr, c_ptr, N, K, A->strides[0], A->strides[1],
                 B->strides[0], B->strides[1], C->strides[0], C->strides[1]};
    pool_for(M, gemv_one_row, &gctx);
    if (g_timing.enabled) {
      g_timing.calls++;
      g_timing.fallback_s += now_s() - t0;
    }
    return;
  }

  const float *bt = get_bt(b_ptr, N, K);
  if (!bt) {
    // Bt cache exhausted: plain GEMV on X100.
    const double t0 = g_timing.enabled ? now_s() : 0.0;
    GemvCtx gctx{a_ptr, b_ptr, c_ptr, N, K, (int64_t)K, 1, (int64_t)N, 1,
                 (int64_t)N, 1};
    pool_for(M, gemv_one_row, &gctx);
    if (g_timing.enabled) {
      g_timing.calls++;
      g_timing.fallback_s += now_s() - t0;
    }
    return;
  }

  const double t0 = g_timing.enabled ? now_s() : 0.0;
  // Load-balance split: measured throughputs are X100 pool (f4 dot) ≈
  // 18-19 GFLOPS and A100 (double dot) ≈ 8.4 GFLOPS, so the A100 share
  // that equalizes finish time is ≈ 8.4/(8.4+18) ≈ 1/3 of the rows.
  const uint32_t a100Rows = useA100 ? (M / 3) : 0;
  const uint32_t x100Rows = M - a100Rows;
  spert::Future f;
  if (a100Rows > 0) {
    DotArgs gA{a_ptr + (size_t)x100Rows * K, bt, c_ptr + (size_t)x100Rows * N,
               a100Rows, N, K};
    const uint32_t tiles =
        (stream->core_count() > a100Rows) ? a100Rows : stream->core_count();
    f = stream->launch(spert::Grid(tiles), a100_dot_tile, gA);
  }
  const double t1 = g_timing.enabled ? now_s() : 0.0;
  DotCtx dctx{a_ptr, bt, c_ptr, N, K};
  pool_for(x100Rows, dot_one_row, &dctx);
  const double t2 = g_timing.enabled ? now_s() : 0.0;
  if (a100Rows > 0)
    f.sync();
  const double t3 = g_timing.enabled ? now_s() : 0.0;
  if (g_timing.enabled) {
    g_timing.calls++;
    g_timing.launch_s += t1 - t0;
    g_timing.x100_s += t2 - t1;
    g_timing.sync_s += t3 - t2;
  }
  // Exit timestamp for launch_gemm duration measurement.
  if (getenv("BGE_M3_A100_TS")) {
    static int tsIdx = 0;
    const int i = tsIdx++;
    fprintf(stderr, "[a100] TS_EXIT call=%d t=%.3f\n", i, now_s());
    fflush(stderr);
  }
}

// C interface ABI referenced by MLIR func.call lowering after
// `llvm-request-c-wrappers`: the module emits a private wrapper
// `@bge_m3_a100_gemm` that packs the unpacked 21-word ABI into three packed
// descriptors and calls `_mlir_ciface_bge_m3_a100_gemm(ptr, ptr, ptr)`.
extern "C" void _mlir_ciface_bge_m3_a100_gemm(Memref2DF32 *A, Memref2DF32 *B,
                                              Memref2DF32 *C) {
  static int call_idx = 0;
  if (call_idx == 0 && getenv("BGE_M3_A100_TIMING")) {
    g_timing.enabled = true;
    atexit(timing_report);
  }
  const int idx = call_idx++;
  // Per-call entry timestamps (BGE_M3_A100_TS=1): the gaps between
  // consecutive entries measure the graph time between offloaded matmuls.
  if (getenv("BGE_M3_A100_TS")) {
    fprintf(stderr, "[a100] TS call=%d t=%.3f\n", idx, now_s());
    fflush(stderr);
  }
  // Per-inference timing report (143 calls per inference): print deltas of
  // the timing accumulators whenever a full inference has completed.
  if (g_timing.enabled && idx % 143 == 142) {
    static double prevLaunch = 0, prevSync = 0, prevX100 = 0, prevHash = 0;
    fprintf(stderr,
            "[a100] INFERENCE %d: launch=%.3fs sync=%.3fs x100=%.3fs "
            "transpose=%.3fs hash=%.3fs\n",
            idx / 143, g_timing.launch_s - prevLaunch,
            g_timing.sync_s - prevSync, g_timing.x100_s - prevX100,
            g_timing.transpose_s, g_timing.hash_s - prevHash);
    prevLaunch = g_timing.launch_s;
    prevSync = g_timing.sync_s;
    prevX100 = g_timing.x100_s;
    prevHash = g_timing.hash_s;
    g_timing.transpose_s = 0;
    fflush(stderr);
  }
  // Diagnostic dump: BGE_M3_A100_DUMP=<n> (or comma-separated list) writes
  // A/B (before) and C (after) of call n to /tmp/mm_dump_{A,B,C}_{n}.bin for
  // offline comparison.
  const bool doDump = [&] {
    const char *dumpEnv = getenv("BGE_M3_A100_DUMP");
    if (!dumpEnv)
      return false;
    const char *p = dumpEnv;
    while (*p) {
      while (*p == ' ' || *p == ',')
        ++p;
      if (*p == '\0')
        break;
      char *end = nullptr;
      const long v = strtol(p, &end, 10);
      if (end == p)
        break;
      if (v == (long)idx)
        return true;
      p = end;
    }
    return false;
  }();
  if (doDump) {
    fprintf(stderr,
            "[a100][%d] DUMP DESC: A(aligned=%p off=%ld sz=%ld,%ld st=%ld,%ld) "
            "B(aligned=%p off=%ld sz=%ld,%ld st=%ld,%ld) "
            "C(aligned=%p off=%ld sz=%ld,%ld st=%ld,%ld)\n",
            idx, (void *)A->aligned, (long)A->offset, (long)A->sizes[0],
            (long)A->sizes[1], (long)A->strides[0], (long)A->strides[1],
            (void *)B->aligned, (long)B->offset, (long)B->sizes[0],
            (long)B->sizes[1], (long)B->strides[0], (long)B->strides[1],
            (void *)C->aligned, (long)C->offset, (long)C->sizes[0],
            (long)C->sizes[1], (long)C->strides[0], (long)C->strides[1]);
    fflush(stderr);
    char path[128];
    snprintf(path, sizeof(path), "/tmp/mm_dump_A_%d.bin", idx);
    FILE *f = fopen(path, "wb");
    if (f) {
      for (int64_t m = 0; m < A->sizes[0]; ++m)
        fwrite(A->aligned + A->offset + m * A->strides[0], sizeof(float),
               (size_t)A->sizes[1], f);
      fclose(f);
    }
    snprintf(path, sizeof(path), "/tmp/mm_dump_B_%d.bin", idx);
    f = fopen(path, "wb");
    if (f) {
      for (int64_t k = 0; k < B->sizes[0]; ++k)
        fwrite(B->aligned + B->offset + k * B->strides[0], sizeof(float),
               (size_t)B->sizes[1], f);
      fclose(f);
    }
    fprintf(stderr, "[a100][%d] DUMPED A/B before compute\n", idx);
    fflush(stderr);
  }
  launch_gemm(A, B, C);
  if (doDump) {
    char path[128];
    snprintf(path, sizeof(path), "/tmp/mm_dump_C_%d.bin", idx);
    FILE *f = fopen(path, "wb");
    if (f) {
      for (int64_t m = 0; m < C->sizes[0]; ++m)
        fwrite(C->aligned + C->offset + m * C->strides[0], sizeof(float),
               (size_t)C->sizes[1], f);
      fclose(f);
    }
    fprintf(stderr, "[a100][%d] DUMPED C after compute\n", idx);
    fflush(stderr);
  }
}
