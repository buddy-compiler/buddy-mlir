#include "gemmini.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// -----------------------------------------------------------------------------
// utils
// -----------------------------------------------------------------------------

static uint64_t read_cycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

// -----------------------------------------------------------------------------
// gemm_malloc from exo
// -----------------------------------------------------------------------------

#ifndef GEMM_HEAP_SIZE
#define GEMM_HEAP_SIZE 100000
#endif

#ifndef GEMM_DIM
#define GEMM_DIM 16
#endif

typedef struct __attribute__((__packed__)) NewBlock {
  uint32_t size;
  uint32_t loc;
  uint8_t is_used;
} NewBlock;

NewBlock BLOCKS[GEMM_HEAP_SIZE / sizeof(NewBlock)];
uint32_t gemm_last_ptr;

void gemm_init_mem() {
  for (uint32_t i = 0; i < sizeof(BLOCKS); i++)
    ((uint8_t *)BLOCKS)[i] = 0;
  gemm_last_ptr = 0;
}

uint32_t gemm_malloc(long unsigned int size) {
  if (size == 0)
    return -1;
  size = (size + GEMM_DIM - 1) / GEMM_DIM;
  int i;
  for (i = 0; i < GEMM_HEAP_SIZE / sizeof(NewBlock) && BLOCKS[i].size > 0;
       i++) {
    if (BLOCKS[i].is_used)
      continue;
    if (BLOCKS[i].size < size)
      continue;
    break;
  }
  if (BLOCKS[i].size == 0) {
    BLOCKS[i].loc = gemm_last_ptr;
    BLOCKS[i].size = size;
    BLOCKS[i].is_used = 1;
    gemm_last_ptr += size;
    return BLOCKS[i].loc;
  }

  BLOCKS[i].is_used = 1;
  return BLOCKS[i].loc;
}

void gemm_free(uint32_t addr) {
  for (int i = 0; BLOCKS[i].size > 0; i++) {
    if (BLOCKS[i].is_used && BLOCKS[i].loc == addr) {
      BLOCKS[i].is_used = 0;
      return;
    }
  }
  return;
}

// -----------------------------------------------------------------------------
// gemm_acc_malloc from exo
// -----------------------------------------------------------------------------

#ifndef GEMM_ACC_HEAP_SIZE
#define GEMM_ACC_HEAP_SIZE 100000
#endif

#ifndef GEMM_ACC_DIM
#define GEMM_ACC_DIM 16
#endif

typedef struct __attribute__((__packed__)) AccBlock {
  uint32_t size;
  uint32_t loc;
  uint8_t is_used;
} AccBlock;

// maintain a stack of blocks corresponding to
// a stack alloc and free strategy
#define N_ACC_BLOCKS (GEMM_ACC_HEAP_SIZE / sizeof(AccBlock))
AccBlock ACC_BLOCKS[N_ACC_BLOCKS];
uint32_t gemm_acc_free_block;

void gemm_acc_init_mem() {
  uint8_t *buf = (uint8_t *)ACC_BLOCKS;
  for (uint32_t i = 0; i < sizeof(ACC_BLOCKS); i++)
    buf[i] = 0;
  gemm_acc_free_block = 0;
}

uint32_t gemm_acc_malloc(long unsigned int size) {
  // must have two free metadata blocks and
  // this allocation must have > 0 size
  if (size == 0)
    return -1;
  if (gemm_acc_free_block >= N_ACC_BLOCKS)
    return -1;

  size = (size + GEMM_ACC_DIM - 1) / GEMM_ACC_DIM;
  uint32_t i = gemm_acc_free_block;

  uint32_t loc = 0;
  if (i > 0) {
    loc = ACC_BLOCKS[i - 1].loc + ACC_BLOCKS[i - 1].size;
  }

  ACC_BLOCKS[i].size = size;
  ACC_BLOCKS[i].loc = loc;
  ACC_BLOCKS[i].is_used = 1;
  gemm_acc_free_block = i + 1;

  return (ACC_BLOCKS[i].loc | ((uint32_t)0x80000000));
}

void gemm_acc_free(uint32_t addr) {
  if (gemm_acc_free_block == 0)
    return;
  addr = addr & (uint32_t)(0x7FFFFFFF);
  // first case: free-ing the top of the block-stack
  if (ACC_BLOCKS[gemm_acc_free_block - 1].loc == addr) {
    ACC_BLOCKS[gemm_acc_free_block - 1].is_used = 0;

    // Then go through and release as many blocks
    // as we can
    for (int i = gemm_acc_free_block - 1; i >= 0; i--) {
      if (ACC_BLOCKS[i].is_used)
        break; // loop termination
      // otherwise...
      gemm_acc_free_block = i;
    }
    // second case: find the freed block and mark it
  } else {
    for (int i = gemm_acc_free_block - 1; i >= 0; i--) {
      if (ACC_BLOCKS[i].loc == addr) {
        ACC_BLOCKS[i].is_used = 0;
        break;
      }
    }
  }
  return;
}

// -----------------------------------------------------------------------------
// exo-generated gemmini kernel
// -----------------------------------------------------------------------------

// matmul_4(
//     scale : f32 @DRAM,
//     act : bool,
//     A : i8[12544, 64] @DRAM,
//     B : i8[64, 256] @DRAM,
//     C : i8[12544, 256] @DRAM
// )

// clang-format off
void matmul_4(const float* scale, bool act, const int8_t* A, const int8_t* B, int8_t* C ) {
  gemmini_extended_config_st((256), (act), (scale)[0]);

  gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);

  gemmini_extended3_config_ld((256), 1.0f, 0, 2);

  gemmini_extended3_config_ld((64), 1.0f, 0, 1);

  gemmini_extended3_config_ld(0, 1.0f, 0, 0);

  int8_t *a = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 1 * 196 * sizeof(int8_t)));
  int8_t *b = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 4 * 1 * 4 * sizeof(int8_t)));
  int32_t *res = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 16 * 4 * 4 * sizeof(int32_t)));
  for (int_fast32_t io = 0; io < 4; io++) {
    for (int_fast32_t i = 0; i < 196; i++) {
      for (int_fast32_t j = 0; j < 4; j++) {
        gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))),(16), (16) );
        gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))),(16), (16) );
        gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))),(16), (16) );
        gemmini_extended_mvin( 0, ((uint64_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))),(16), (16) );
        if (j == 0) {
          gemmini_extended_mvin2( &A[(16 * i + 3136 * io) * (64)], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), 16*(4), (16) );
        }
        if (io == 0) {
          if (i == 0) {
            gemmini_extended_mvin3( &B[64 * j], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096))/16))), 16*(4), (16) );
          }
        }
        if (io == 0) {
          if (i == 0) {
            gemmini_extended_mvin3( &B[(16) * (256) + 64 * j], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024)/16))), 16*(4), (16) );
          }
        }
        if (io == 0) {
          if (i == 0) {
            gemmini_extended_mvin3( &B[(32) * (256) + 64 * j], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024))/16))), 16*(4), (16) );
          }
        }
        if (io == 0) {
          if (i == 0) {
            gemmini_extended_mvin3( &B[(48) * (256) + 64 * j], ((uint64_t) &*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024))/16))), 16*(4), (16) );
          }
        }
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024 + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024 + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + 1024 + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + 256)/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (2) * (1024) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (2) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024) + 256)/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024) + (2) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_preload((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)b)) + ((j) * (4096) + (3) * (1024) + (3) * (256))/16))), (uint32_t)(&*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16))) | 0x40000000, (16), (16), (16), (16));
        gemmini_extended_compute_preloaded((uint32_t)(&*(int8_t*)((uint64_t)( ((uint32_t)((uint64_t)a)) + ((i) * (1024) + (3) * (256))/16))), ~((uint32_t)0), (16), (16), 16, 16);
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16)), (16), (16) );
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 16 + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16)), (16), (16) );
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 32 + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16)), (16), (16) );
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 48 + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16)), (16), (16) );
      }
    }
  }
  gemm_acc_free((uint32_t)(res));
  gemm_free((uint64_t)(b));
  gemm_free((uint64_t)(a));
}
// clang-format on

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
#define MM 256
#define NN 12544
#define KK 64

static float c_scale[1] = {2.0f};
static int8_t x[NN * KK];
static int8_t y[KK * MM];
static int8_t z_cpu[NN * MM] = {0};
static int8_t z_gemmini[NN * MM] = {0};

int main() {
  gemm_init_mem();
  gemm_acc_init_mem();
  gemmini_flush(0);

  for (int i = 0; i < NN; i++) {
    for (int j = 0; j < KK; j++) {
      x[(KK)*i + j] = i + j * 2;
    }
  }

  for (int i = 0; i < KK; i++) {
    for (int j = 0; j < MM; j++) {
      y[(MM)*i + j] = j * 3 + i;
    }
  }

  unsigned long gemmini_start = read_cycles();
  matmul_4(c_scale, false, x, y, z_gemmini);
  gemmini_fence();
  unsigned long gemmini_stop = read_cycles();
  printf("Cycles for GEMMINI version: %ld\n", gemmini_stop - gemmini_start);
  printf("\nDone\n");

  exit(0);
}
