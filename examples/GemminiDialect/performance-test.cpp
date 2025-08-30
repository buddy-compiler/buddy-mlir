#include <buddy/Core/Container.h>

using namespace std;

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
#define I 1
#define K 1
#define J 1
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

// If DIALECT is 1,we use linalg dialect,otherwise we use gemmini.
#ifndef DIALECT 
#define DIALECT 0
#endif

extern "C" {
void _mlir_ciface_linalg_matmul1(MemRef<int8_t, 2> *input0, MemRef<int8_t, 2> *input1,
                          MemRef<int8_t, 2> *output);

void _mlir_ciface_linalg_matmul2(MemRef<int8_t, 2> *input0, MemRef<int8_t, 2> *input1,
                          MemRef<int8_t, 2> *output);

void _mlir_ciface_linalg_matmul3(MemRef<int8_t, 2> *input0, MemRef<int8_t, 2> *input1,
                          MemRef<int8_t, 2> *output);

void _mlir_ciface_linalg_matmul4(MemRef<int8_t, 2> *input0, MemRef<int8_t, 2> *input1,
                          MemRef<int8_t, 2> *output);

void _mlir_ciface_linalg_matmul5(MemRef<int8_t, 2> *input0, MemRef<int8_t, 2> *input1,
                          MemRef<int8_t, 2> *output);

void _mlir_ciface_linalg_matmul6(MemRef<int8_t, 2> *input0, MemRef<int8_t, 2> *input1,
                          MemRef<int8_t, 2> *output);


void _mlir_ciface_gemmini_matmul1(MemRef<int8_t, 2> *input0,
                                  MemRef<int8_t, 2>  *input1,
                                  MemRef<int8_t, 2>  *output ,
                                  MemRef<int32_t, 2> *bias);

void _mlir_ciface_gemmini_matmul2(MemRef<int8_t, 2> *input0,
                                  MemRef<int8_t, 2>  *input1,
                                  MemRef<int8_t, 2>  *output ,
                                  MemRef<int32_t, 2> *bias);

void _mlir_ciface_gemmini_matmul3(MemRef<int8_t, 2> *input0,
                                  MemRef<int8_t, 2>  *input1,
                                  MemRef<int8_t, 2>  *output ,
                                   MemRef<int32_t, 2> *bias);

void _mlir_ciface_gemmini_matmul4(MemRef<int8_t, 2> *input0,
                                  MemRef<int8_t, 2>  *input1,
                                  MemRef<int8_t, 2>  *output ,
                                  MemRef<int32_t, 2> *bias);


void _mlir_ciface_gemmini_matmul5(MemRef<int8_t, 2> *input0,
                                  MemRef<int8_t, 2>  *input1,
                                  MemRef<int8_t, 2>  *output ,
                                  MemRef<int32_t, 2> *bias);

void _mlir_ciface_gemmini_matmul6(MemRef<int8_t, 2> *input0,
                                 MemRef<int8_t, 2>  *input1,
                                 MemRef<int8_t, 2>  *output ,
                                 MemRef<int32_t, 2> *bias);

void _mlir_ciface_linalg_conv1(MemRef<int8_t, 4> *input,
                                     MemRef<int8_t, 4> *kernel,
                                     MemRef<int8_t, 4> *output);

void _mlir_ciface_linalg_conv2(MemRef<int8_t, 4> *input,
                                     MemRef<int8_t, 4> *kernel,
                                     MemRef<int8_t, 4> *output);

void _mlir_ciface_linalg_conv3(MemRef<int8_t, 4> *input,
                                     MemRef<int8_t, 4> *kernel,
                                     MemRef<int8_t, 4> *output);

void _mlir_ciface_linalg_conv4(MemRef<int8_t, 4> *input,
                                     MemRef<int8_t, 4> *kernel,
                                     MemRef<int8_t, 4> *output);

void _mlir_ciface_linalg_conv5(MemRef<int8_t, 4> *input,
                                     MemRef<int8_t, 4> *kernel,
                                     MemRef<int8_t, 4> *output);

void _mlir_ciface_linalg_conv6(MemRef<int8_t, 4> *input,
                                     MemRef<int8_t, 4> *kernel,
                                     MemRef<int8_t, 4> *output);

void _mlir_ciface_gemmini_conv1(MemRef<int8_t, 4>* input,
                                MemRef<int8_t, 2>* weights,
                                MemRef<int32_t, 1>* bias, 
                                MemRef<int8_t, 2> *output);

void _mlir_ciface_gemmini_conv2(MemRef<int8_t, 4>* input,
                                MemRef<int8_t, 2>* weights,
                                MemRef<int32_t, 1>* bias, 
                                MemRef<int8_t, 2> *output);

void _mlir_ciface_gemmini_conv3(MemRef<int8_t, 4>* input,
                                MemRef<int8_t, 2>* weights,
                                MemRef<int32_t, 1>* bias, 
                                MemRef<int8_t, 2> *output);

void _mlir_ciface_gemmini_conv4(MemRef<int8_t, 4>* input,
                                MemRef<int8_t, 2>* weights,
                                MemRef<int32_t, 1>* bias, 
                                MemRef<int8_t, 2> *output);

void _mlir_ciface_gemmini_conv5(MemRef<int8_t, 4>* input,
                                MemRef<int8_t, 2>* weights,
                                MemRef<int32_t, 1>* bias, 
                                MemRef<int8_t, 2> *output);
void _mlir_ciface_gemmini_conv6(MemRef<int8_t, 4>* input,
                                MemRef<int8_t, 2>* weights,
                                MemRef<int32_t, 1>* bias, 
                                MemRef<int8_t, 2> *output);

static uint64_t readCycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}
}

int main() {
  if (MATMUL) {
    if (DIALECT == 1) {
      vector<size_t> sizes = {I, K};
      MemRef<int8_t, 2> input0(sizes, 1);
      sizes.assign({K, J});
      MemRef<int8_t, 2> input1(sizes, 2);
      sizes.assign({I, J});
      MemRef<int8_t, 2> output(sizes, 0);
      uint64_t start, end;
      switch (MATMUL) {
      case 1:
        start = readCycles();
        _mlir_ciface_linalg_matmul1(&input0, &input1, &output);
        end = readCycles();
        break;
      case 2:
        start = readCycles();
        _mlir_ciface_linalg_matmul2(&input0, &input1, &output);
        end = readCycles();
        break;
      case 3:
        start = readCycles();
        _mlir_ciface_linalg_matmul3(&input0, &input1, &output);
        end = readCycles();
        break;
      case 4:
        start = readCycles();
        _mlir_ciface_linalg_matmul4(&input0, &input1, &output);
        end = readCycles();
        break;
      case 5:
        start = readCycles();
        _mlir_ciface_linalg_matmul5(&input0, &input1, &output);
        end = readCycles();
        break;
      case 6:
        start = readCycles();
        _mlir_ciface_linalg_matmul6(&input0, &input1, &output);
        end = readCycles();
        break;
      default:
        printf("You specify the wrong matmul test case.\n");
        return 0;
      }
      printf("The linalg.matmul test case is %d\n", MATMUL);
      printf("I = %d K = %d J = %d\n", I, K, J);
      printf("Cycles taken %lld\n", end - start); 
      return 0;
    } else if (DIALECT == 2) {
      vector<size_t> sizes = {I, K}; 
      MemRef<int8_t, 2> input0(sizes, 1);
      MemRef<int8_t, 2> input1(sizes, 2);
      MemRef<int8_t, 2> output(sizes, 0);
      MemRef<int32_t, 2> bias(sizes, 0);
      uint64_t start, end;
      switch (MATMUL) {
      case 1:
        start = readCycles();
        _mlir_ciface_gemmini_matmul1(&input0, &input1, &output, &bias);
        end = readCycles();
        break;
      case 2:
        start = readCycles();
        _mlir_ciface_gemmini_matmul2(&input0, &input1, &output, &bias);
        end = readCycles();
        break;
      case 3:
        start = readCycles();
        _mlir_ciface_gemmini_matmul3(&input0, &input1, &output, &bias);
        end = readCycles();
        break;
      case 4:
        start = readCycles();
        _mlir_ciface_gemmini_matmul4(&input0, &input1, &output, &bias);
        end = readCycles();
        break;
      case 5:
        start = readCycles();
        _mlir_ciface_gemmini_matmul5(&input0, &input1, &output, &bias);
        end = readCycles();
        break;
      case 6:
        start = readCycles();
        _mlir_ciface_gemmini_matmul6(&input0, &input1, &output, &bias);
        end = readCycles();
        break;
      default:
        printf("You specify the wrong matmul test case.\n");
        return 0;
      }
      printf("The gemmini.matmul test case is %d\n", MATMUL);
      printf("I = %d K = %d J = %d\n", I, K, J);
      printf("Cycles taken %lld\n", end - start);
      return 0;
    }
  }

  if (CONV) {
    if (DIALECT == 1) {
      vector<size_t> sizes = {BATCH_SIZE, IN_CHANNELS, IN_DIM, IN_DIM};
      MemRef<int8_t, 4> input(sizes, 1);
      sizes.assign({OUT_CHANNELS, IN_CHANNELS, KERNEL_DIM, KERNEL_DIM});
      MemRef<int8_t, 4> weights(sizes, 1);
      sizes.assign({BATCH_SIZE, OUT_CHANNELS, OUT_DIM, OUT_DIM});
      MemRef<int8_t, 4> output(sizes, 0);
      uint64_t start, end;
      switch (CONV) {
      case 1:
        start = readCycles();
        _mlir_ciface_linalg_conv1(&input, &weights, &output);
        end = readCycles();
        break;
      case 2:
        start = readCycles();
        _mlir_ciface_linalg_conv2(&input, &weights, &output);
        end = readCycles();
        break;
      case 3:
        start = readCycles();
        _mlir_ciface_linalg_conv3(&input, &weights, &output);
        end = readCycles();
        break;
      case 4:
        start = readCycles();
        _mlir_ciface_linalg_conv4(&input, &weights, &output);
        end = readCycles();
        break;
      case 5:
        start = readCycles();
        _mlir_ciface_linalg_conv5(&input, &weights, &output);
        end = readCycles();
        break;
      case 6:
        start = readCycles();
        _mlir_ciface_linalg_conv6(&input, &weights, &output);
        end = readCycles();
        break;
      default:
        printf("You specify the wrong conv test case.\n");
        return 0;
      }
      printf("The linalg.conv test case is %d\n", CONV);
      printf("BATCH_SIZE = %d IN_CHANNELS = %d OUT_CHANNELS = %d IN_DIM = %d "
            "KERNEL_DIM = %d OUT_DIM = %d\n",
            BATCH_SIZE, IN_CHANNELS, OUT_CHANNELS, IN_DIM, KERNEL_DIM, OUT_DIM);
      printf("Cycles taken = %lld\n", end - start);
      return 0;
    } else if(DIALECT == 2) {
      vector<size_t> sizes = {BATCH_SIZE, IN_DIM, IN_DIM, IN_CHANNELS};
      MemRef<int8_t, 4> input(sizes, 1); 
      sizes.assign({KERNEL_DIM * KERNEL_DIM, 1}); 
      MemRef<int8_t, 2> weights(sizes, 1); 
      sizes.assign({OUT_CHANNELS}); 
      MemRef<int32_t, 1>  bias(sizes, 0); 
      sizes.assign({OUT_DIM * OUT_DIM, 1});
      MemRef<int8_t, 2> output(sizes, 0);
      uint64_t start, end;
      switch (CONV) {
      case 1:
        start = readCycles();
        _mlir_ciface_gemmini_conv1(&input, &weights, &bias, &output);
        end = readCycles();
        break;
      case 2:
        start = readCycles();
        _mlir_ciface_gemmini_conv2(&input, &weights, &bias, &output);
        end = readCycles();
        break;
      case 3:
        start = readCycles();
        _mlir_ciface_gemmini_conv3(&input, &weights, &bias, &output);
        end = readCycles();
        break;
      case 4:
        start = readCycles();
        _mlir_ciface_gemmini_conv4(&input, &weights, &bias, &output);
        end = readCycles();
        break;
      case 5:
        start = readCycles();
        _mlir_ciface_gemmini_conv5(&input, &weights, &bias, &output);
        end = readCycles();
        break;
      case 6:
        start = readCycles();
        _mlir_ciface_gemmini_conv6(&input, &weights, &bias, &output);
        end = readCycles();
        break;
      default:
        printf("You specify the wrong conv test case.\n");
        return 0;
      }
      printf("The gemmini.conv test case is %d\n", CONV);
      printf("BATCH_SIZE = %d IN_CHANNELS = %d OUT_CHANNELS = %d IN_DIM = %d "
            "KERNEL_DIM = %d OUT_DIM = %d\n",
            BATCH_SIZE, IN_CHANNELS, OUT_CHANNELS, IN_DIM, KERNEL_DIM, OUT_DIM);
      printf("Cycles taken = %lld\n", end - start);
      return 0;
    }
  }
}
