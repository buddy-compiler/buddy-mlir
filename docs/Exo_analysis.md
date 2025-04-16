<!--
Copyright (c) 2025 <Jasper.zeng>

SPDX-License-Identifier: Apache-2.0
-->

# Introduce
This document presents the analytical process and conclusions regarding the use of Exo-lang on Gemmini hardware. We will focus on the following points:

## 1. Understanding the Code in *exo-matmul-4.c*:
```c
 int8_t *a = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 1 * 196 * sizeof(int8_t)));
  int8_t *b = (int8_t*) ((uint64_t)gemm_malloc (16 * 16 * 4 * 4 * 1 * 4 * sizeof(int8_t)));
  int32_t *res = (int32_t*) ((uint32_t)gemm_acc_malloc (16 * 16 * 4 * 4 * sizeof(int32_t)));
```

>Questions to address:
>- What do these data types mean?
>- Where do the numbers like 16 and 196 originate from?

## 2. Kernel Shape Modification in *exo-matmul-4.c*
```c
#define MM 64 //256 -> 64
#define NN 12544
#define KK 64
```
> Observations:
>- Changing MM from 256 to 64 and running the simulation on Spike results in a crash with the following error:

the error information from spike and this indicates a "memory segmentation fault error."
```bash
bbl loader
Gemmini extension configured with:
    dim = 16
z  0000000000000000 ra 000000000025b8e8 sp 0000003ffffff840 gp 000000000007c098
tp 000000000023b140 t0 0010001000001620 t1 00100010c0000000 t2 0000000000000000
s0 0010001000001610 s1 0010001000001600 a0 00100010c0000030 a1 0010001080000000
a2 0000000080000010 a3 0000000080000020 a4 0000000080000030 a5 0000000000000000
a6 00100010c0000020 a7 00100010c0000010 s2 0000000080000000 s3 0000000000003100
s4 0000000040000000 s5 0000000080000000 s6 ffffffff80000000 s7 00100010ffffffff
s8 0010001000000000 s9 0000000000003100 sA 00000000ffffffff sB 0000000000000000
t3 00100010000031c0 t4 0010001000003180 t5 0010001000003140 t6 0010001000001630
pc 0000000000010d68 va/inst 000000000025c0e8 sr 8000000200006020
User store segfault @ 0x000000000025c0e8
make: *** [makefile:2491: exo-matmul-4-run] Error 255
```


## 3.Testing and optimizing the Exo-matmul Kernel:

>We have observed that Exo-lang for Gemmini is not as fast as the Buddy-compiler. We aim to use Exo-lang as a benchmark for the Buddy-compiler. Therefore, it is crucial to determine whether the Exo-matmul-4 kernel achieves its expected performance.

# Analysis
## Q1 Understanding the Code:
I found that it is easy to print the modifidation by exo-lang just using:
print(gemmini) in exo-lang, such as: we could find the source *matmul.py* files in exo project: *exo/apps/gemmini/src/exo/matmul.py* then we just add the print(gemmini) after where we want to see, like:
```py
def sched_matmul(
    name,
    NN,
    MM,
    KK,
):
    cpu = rename(matmul_on_cpu, f"cpu_{name}")
    cpu = cpu.partial_eval(NN, MM, KK)

    gemmini = rename(cpu, name)
    print("\n=== Stage 1: Initial state after rename ===")
    print(gemmini)

    gemmini = set_memory(gemmini, "res", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "a", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "b", GEMM_SCRATCH)
    """
    modify the res, a, b to GEMM's inside Mem
    """
    print("\n=== Stage 2: After setting memory types ===")
    print(gemmini)
```
then we run this python file, we can get the modify such as:
```py
=== Stage 1: Initial state after rename ===
def matmul_4(scale: f32 @ DRAM, act: bool @ DRAM, A: i8[12544, 64] @ DRAM,
             B: i8[64, 256] @ DRAM, C: i8[12544, 256] @ DRAM):
    for i in seq(0, 12544):
        for j in seq(0, 256):
            res: i32 @ DRAM
            res = 0.0
            for k in seq(0, 64):
                a: i8 @ DRAM
                a = A[i, k]
                b: i8 @ DRAM
                b = B[k, j]
                a2: i32 @ DRAM
                b2: i32 @ DRAM
                a2 = a
                b2 = b
                res += a2 * b2
            src_tmp: i32 @ DRAM
            src_tmp = res
            tmp_res1: f32 @ DRAM
            acc_scale(src_tmp, tmp_res1, scale)
            tmp_res2: i8 @ DRAM
            clamp(tmp_res1, tmp_res2)
            if act == True:
                tmp_res2 = relu(tmp_res2)
            C[i, j] = tmp_res2

=== Stage 2: After setting memory types ===
def matmul_4(scale: f32 @ DRAM, act: bool @ DRAM, A: i8[12544, 64] @ DRAM,
             B: i8[64, 256] @ DRAM, C: i8[12544, 256] @ DRAM):
    for i in seq(0, 12544):
        for j in seq(0, 256):
            res: i32 @ GEMM_ACCUM
            res = 0.0
            for k in seq(0, 64):
                a: i8 @ GEMM_SCRATCH
                a = A[i, k]
                b: i8 @ GEMM_SCRATCH
                b = B[k, j]
                a2: i32 @ DRAM
                b2: i32 @ DRAM
                a2 = a
                b2 = b
                res += a2 * b2
            src_tmp: i32 @ DRAM
            src_tmp = res
            tmp_res1: f32 @ DRAM
            acc_scale(src_tmp, tmp_res1, scale)
            tmp_res2: i8 @ DRAM
            clamp(tmp_res1, tmp_res2)
            if act == True:
                tmp_res2 = relu(tmp_res2)
            C[i, j] = tmp_res2
```
Thanks to this special instruction, we can easily identify the differences at each step. To visualize these changes more clearly, similar to a git diff, I wrote a program. To use this tool, follow these steps:
1. Insert the following code at the desired location in your original Python file:
```py
    print("\n=== Stage N: description ===")
     # This will search based on the pattern === Stage N
    print(gemmini)
```
2. Ensure that N is incrementing and unique; it can be a floating-point number.
3. Run the program to automatically generate an HTML file, which can be opened in a web browser or with a VSCode HTML plugin.
```bash
python matmul.py > name.log
python analyze_rewrite.py name.log

#get: name_diff_report.html
``` 
Finally, I obtained the modified version:
```python
def matmul_4(scale: f32 @ DRAM, act: bool @ DRAM, A: i8[12544, 64] @ DRAM,
             B: i8[64, 256] @ DRAM, C: i8[12544, 256] @ DRAM):
    config_st_acc_i8(scale, stride(C, 0), act)
    config_matmul()
    config_ld_i8_id2(stride(B, 0))
    config_ld_i8_id1(stride(A, 0))
    config_zero()
    a: i8[196, 1, 4, 16, 16] @ GEMM_SCRATCH
    b: i8[4, 1, 4, 4, 16, 16] @ GEMM_SCRATCH
    res: i32[4, 4, 16, 16] @ GEMM_ACCUM
    for io in seq(0, 4):
        for i in seq(0, 196):
            for j in seq(0, 4):
                do_zero_acc_i32(16, 16, res[j, 0, 0:16, 0:16])
                do_zero_acc_i32(16, 16, res[j, 1, 0:16, 0:16])
                do_zero_acc_i32(16, 16, res[j, 2, 0:16, 0:16])
                do_zero_acc_i32(16, 16, res[j, 3, 0:16, 0:16])
                if j == 0:
                    do_ld_i8_block_id1(
                        16, 4, A[16 * i + 3136 * io:16 + 16 * i + 3136 * io,
                                 0:64], a[i, 0, 0:4, 0:16, 0:16])
                if io == 0:
                    if i == 0:
                        do_ld_i8_block_id2(
                            16, 4, B[16 * 0:16 + 16 * 0, 64 * j:64 + 64 * j],
                            b[j, 0, 0, 0:4, 0:16, 0:16])
                if io == 0:
                    if i == 0:
                        do_ld_i8_block_id2(
                            16, 4, B[16 * 1:16 + 16 * 1, 64 * j:64 + 64 * j],
                            b[j, 0, 1, 0:4, 0:16, 0:16])
                if io == 0:
                    if i == 0:
                        do_ld_i8_block_id2(
                            16, 4, B[16 * 2:16 + 16 * 2, 64 * j:64 + 64 * j],
                            b[j, 0, 2, 0:4, 0:16, 0:16])
                if io == 0:
                    if i == 0:
                        do_ld_i8_block_id2(
                            16, 4, B[16 * 3:16 + 16 * 3, 64 * j:64 + 64 * j],
                            b[j, 0, 3, 0:4, 0:16, 0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 0, 0:16, 0:16],
                                 b[j, 0, 0, 0, 0:16, 0:16], res[j, 0, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 0, 0:16, 0:16],
                                 b[j, 0, 0, 1, 0:16, 0:16], res[j, 1, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 0, 0:16, 0:16],
                                 b[j, 0, 0, 2, 0:16, 0:16], res[j, 2, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 0, 0:16, 0:16],
                                 b[j, 0, 0, 3, 0:16, 0:16], res[j, 3, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 1, 0:16, 0:16],
                                 b[j, 0, 1, 0, 0:16, 0:16], res[j, 0, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 1, 0:16, 0:16],
                                 b[j, 0, 1, 1, 0:16, 0:16], res[j, 1, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 1, 0:16, 0:16],
                                 b[j, 0, 1, 2, 0:16, 0:16], res[j, 2, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 1, 0:16, 0:16],
                                 b[j, 0, 1, 3, 0:16, 0:16], res[j, 3, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 2, 0:16, 0:16],
                                 b[j, 0, 2, 0, 0:16, 0:16], res[j, 0, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 2, 0:16, 0:16],
                                 b[j, 0, 2, 1, 0:16, 0:16], res[j, 1, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 2, 0:16, 0:16],
                                 b[j, 0, 2, 2, 0:16, 0:16], res[j, 2, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 2, 0:16, 0:16],
                                 b[j, 0, 2, 3, 0:16, 0:16], res[j, 3, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 3, 0:16, 0:16],
                                 b[j, 0, 3, 0, 0:16, 0:16], res[j, 0, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 3, 0:16, 0:16],
                                 b[j, 0, 3, 1, 0:16, 0:16], res[j, 1, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 3, 0:16, 0:16],
                                 b[j, 0, 3, 2, 0:16, 0:16], res[j, 2, 0:16,
                                                                0:16])
                do_matmul_acc_i8(16, 16, 16, a[i, 0, 3, 0:16, 0:16],
                                 b[j, 0, 3, 3, 0:16, 0:16], res[j, 3, 0:16,
                                                                0:16])
                do_st_acc_i8(
                    16, 16, res[j, 0, 0:16, 0:16],
                    C[16 * i + 3136 * io:16 + 16 * i + 3136 * io,
                      16 * 0 + 64 * j:16 + 16 * 0 + 64 * j])
                do_st_acc_i8(
                    16, 16, res[j, 1, 0:16, 0:16],
                    C[16 * i + 3136 * io:16 + 16 * i + 3136 * io,
                      16 * 1 + 64 * j:16 + 16 * 1 + 64 * j])
                do_st_acc_i8(
                    16, 16, res[j, 2, 0:16, 0:16],
                    C[16 * i + 3136 * io:16 + 16 * i + 3136 * io,
                      16 * 2 + 64 * j:16 + 16 * 2 + 64 * j])
                do_st_acc_i8(
                    16, 16, res[j, 3, 0:16, 0:16],
                    C[16 * i + 3136 * io:16 + 16 * i + 3136 * io,
                      16 * 3 + 64 * j:16 + 16 * 3 + 64 * j])
```
This version is highly similar to the kernel in exo-matmul-4.c. Additionally, through the intermediate steps, we can clearly understand how the parameters in the kernel are generated.



## Q2 Kernel Shape Modification Error
Using Spike and GDB, I identified that the  ***User store segfault @ 0x000000000025c0e8***  corresponds to：*line-251 in exo-matmul-4.c*
```c
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16)), (16), (16) );
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 16 + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + 256)/16)), (16), (16) );
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 32 + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (2) * (256))/16)), (16), (16) );
        gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 48 + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024) + (3) * (256))/16)), (16), (16) );
```
Further analysis reveals that changing MM from 256 to 64 in the original code causes a memory access error. The error occurs during the gemmini_extended_mvout function call, which is a memory write operation.

> **Memory Allocation Issue:**<br>
> The z_gemmini array is defined as static int8_t z_gemmini[NN * MM]. When MM = 64, the array size is 12544 * 64. However, the access expression still uses 256 as a multiplier: (16 * i + 3136 * io) * (256) + 64 * j.

> **Out-of-Bounds Access Analysis:**<br>
> The expression (16 * i + 3136 * io) * (256) + 64 * j uses a hardcoded 256. Changing MM to 64 causes this expression to access beyond the actual allocated size of the z_gemmini array. The array is allocated for MM=64, but the access pattern calculates offsets as if MM=256.

>**Fix Recommendation:**<br>
>Modify the access expression in gemmini_extended_mvout to replace the hardcoded 256 with MM. The expression should be: C[(16 * i + 3136 * io) * (MM) + 64 * j]. 

**In summary, the values of MM, NN, and KK are related to the parameters and the number of times devide_loop is configured in Exo. Additionally, address out-of-bounds access issues must be considered.**

## Q3 optimizing the Exo-matmul Kernel
In the function call to gemmini_extended_mvout, the parameters need to be carefully considered to ensure they align with the intended memory layout and dimensions. Let's break down the parameters:
```c
gemmini_extended_mvout( ((uint64_t) &C[(16 * i + 3136 * io) * (256) + 64 * j]), (uint32_t) &*(int32_t*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16)), (16), (16) );
```
**Destination Address:**
((uint64_t) &C[(16 * i + 3136 * io) * (256) + 64 * j])
The expression (16 * i + 3136 * io) * (256) + 64 * j calculates the offset into the C array.
The hardcoded 256 should be replaced with MM to ensure it matches the current matrix dimension. This is crucial to prevent out-of-bounds access.

**Source Address:**
(uint32_t) &\*(int32_t\*)((uint64_t)( ((uint32_t)((uint64_t)res)) + ((j) * (1024))/16))
This calculates the source address in the res array. Ensure that the calculation of the offset (j) * (1024)/16 is correct and consistent with the intended data layout.