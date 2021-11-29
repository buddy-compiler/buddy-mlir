//====- kernels.h - Kernels definition  -----------------------------------===//
//
// This file defines the kernel of the edge detection examples.
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLES_CONV_OPT_KERNELS
#define EXAMPLES_CONV_OPT_KERNELS

float prewittKernelAlign[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
int prewittKernelRows = 3;
int prewittKernelCols = 3;

float sobel3x3KernelAlign[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
int sobel3x3KernelRows = 3;
int sobel3x3KernelCols = 3;

float sobel5x5KernelAlign[25] = {2, 1, 0, -1, -2,
                                 3, 2, 0, -2, -3,
                                 4, 3, 0, -3, -4,
                                 3, 2, 0, -2, -3,
                                 2, 1, 0, -1, -2};
int sobel5x5KernelRows = 5;
int sobel5x5KernelCols = 5;

float sobel7x7KernelAlign[49] = {3, 2, 1, 0, -1, -2, -3,
                                 4, 3, 2, 0, -2, -3, -4,
                                 5, 4, 3, 0, -3, -4, -5,
                                 6, 5, 4, 0, -4, -5, -6,
                                 5, 4, 3, 0, -3, -4, -5,
                                 4, 3, 2, 0, -2, -3, -4,
                                 3, 2, 1, 0, -1, -2, -3};
int sobel7x7KernelRows = 7;
int sobel7x7KernelCols = 7;

float sobel9x9KernelAlign[81] = {4, 3, 2, 1, 0, -1, -2, -3, -4,
                                 5, 4, 3, 2, 0, -2, -3, -4, -5,
                                 6, 5, 4, 3, 0, -3, -4, -5, -6,
                                 7, 6, 5, 4, 0, -4, -5, -6, -7,
                                 8, 7, 6, 5, 0, -5, -6, -7, -8,
                                 7, 6, 5, 4, 0, -4, -5, -6, -7,
                                 6, 5, 4, 3, 0, -3, -4, -5, -6,
                                 5, 4, 3, 2, 0, -2, -3, -4, -5,
                                 4, 3, 2, 1, 0, -1, -2, -3, -4};
int sobel9x9KernelRows = 9;
int sobel9x9KernelCols = 9;

float laplacianKernelAlign[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
int laplacianKernelRows = 3;
int laplacianKernelCols = 3;

float logKernelAlign[25] = {0, 0, 1, 0, 0, 
                            0, 1, 2, 1, 0,
                            1, 2, -16, 2, 1,
                            0, 1, 2, 1, 0,
                            0, 0, 1, 0, 0};
int logKernelRows = 5;
int logKernelCols = 5;

#endif
