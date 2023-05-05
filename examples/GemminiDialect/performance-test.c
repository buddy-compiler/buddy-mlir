#include "gemmini.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef MATMUL

#if MATMUL == 1
#define I 32
#define K 32
#define J 32

#elif MATMUL == 2
#define I 64
#define J 64
#define K 64

#elif MATMUL == 3
#define I 128
#define J 128
#define K 128

#elif MATMUL == 4
#define I 256
#define J 256
#define K 256

#elif MATMUL == 5
#define I 512
#define J 512
#define K 512

#elif MATMUL == 6
#define I 1024
#define J 1024 
#define K 1024

#endif
#endif

#ifndef MATMUL
#define MATMUL 0
#define I 0
#define K 0
#define J 0
#endif

#ifdef CONV
#define BATCH_SIZE 1
#define IN_CHANNELS 1
#define OUT_CHANNELS 1
#define IN_DIM 256
#if CONV == 1
#define KERNEL_DIM 3
#define OUT_DIM 254

#elif CONV == 2
#define KERNEL_DIM 5
#define OUT_DIM 252

#elif CONV == 3
#define KERNEL_DIM 7 
#define OUT_DIM 250

#elif CONV == 4
#define KERNEL_DIM 9 
#define OUT_DIM 248

#elif CONV == 5
#define KERNEL_DIM 11 
#define OUT_DIM 246 

#elif CONV == 6 
#define KERNEL_DIM 13 
#define OUT_DIM 244
#endif
#endif

#ifndef CONV
#define CONV 0
#define BATCH_SIZE 0
#define IN_CHANNELS 0
#define OUT_CHANNELS 0
#define KERNEL_DIM 0
#define IN_DIM 0
#define OUT_DIM 0
#endif

static uint64_t read_cycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

int main() {
  if (MATMUL) {
    static elem_t a[I][K];
    static elem_t b[K][J];
    static elem_t c[I][J];
    
    for (size_t i = 0; i != I; i++)
      for (size_t k = 0; k != K; k++)
        a[i][k] = 1;
    static acc_t d[I][J];
    for (size_t k = 0; k != K; k++)
      for (size_t j = 0; j != J; j++)
        b[k][j] = 2;

    uint64_t start = read_cycles();
    tiled_matmul_auto(I, J, K, (elem_t *)a, (elem_t *)b, NULL, (elem_t *)c,
                      K, J, J, J, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                      MVIN_SCALE_IDENTITY, NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
                      false, false, false, false, 0, 0, WS);
    uint64_t end = read_cycles();
    printf("Test case is %d\n", MATMUL);
    printf("I = %d K = %d J = %d\n", I, J, K);
    printf("Cycles taken %d\n", end - start);
  }

  if (CONV) {
    static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][IN_CHANNELS];
    static elem_t weights[IN_CHANNELS * KERNEL_DIM * KERNEL_DIM][OUT_CHANNELS];
    static acc_t bias[OUT_CHANNELS];
    static elem_t output[BATCH_SIZE * OUT_DIM * OUT_DIM][OUT_CHANNELS];
    for (size_t n = 0; n < BATCH_SIZE; n++)
      for (size_t h = 0; h < IN_DIM; h++)
        for (size_t w = 0; w < IN_DIM; w++)
          for (size_t c = 0; c < IN_CHANNELS; c++)
            input[n][h][w][c] = 1;

    for (size_t i = 0; i != IN_CHANNELS * KERNEL_DIM * KERNEL_DIM; i++)
      for (size_t j = 0; j < OUT_CHANNELS; j++)
        weights[i][j] = 1;

    uint64_t start = read_cycles();
    tiled_conv_auto(BATCH_SIZE, IN_DIM, IN_CHANNELS, OUT_CHANNELS, OUT_DIM,
                    /*stride=*/1, 1, 1, /*padding=*/0, KERNEL_DIM, false, false,
                    false, false, false, (elem_t *)input, (elem_t *)weights,
                    (acc_t *)bias, (elem_t *)output, NO_ACTIVATION,
                    ACC_SCALE_IDENTITY, 0, 0, 0, WS);
    uint64_t end = read_cycles();
    printf("Cycles taken = %d\n", end - start);
    return 0;
  }
}
