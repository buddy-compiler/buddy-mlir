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

Here, we could list partitial instructions:

- Config
```python
#python Code
config_st_acc_i8(scale, stride(C, 0), act)
config_matmul()
config_ld_i8_id2(stride(B, 0))
config_ld_i8_id1(stride(A, 0))
config_zero()
```
```c
//C Code
gemmini_extended_config_st((256), (act), (scale)[0]);  // config memory unit
gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);        // config executor
gemmini_extended3_config_ld((256), 1.0f, 0, 2);       // config load of Matrix B
gemmini_extended3_config_ld((64), 1.0f, 0, 1);        // config load of Matrix A
gemmini_extended3_config_ld(0, 1.0f, 0, 0);            //Set Zero
```

------------------------------------
------------------------------------
- Memory Allocation
```py
#python
a: i8[196, 1, 4, 16, 16] @ GEMM_SCRATCH
b: i8[4, 1, 4, 4, 16, 16] @ GEMM_SCRATCH
res: i32[4, 4, 16, 16] @ GEMM_ACCUM
```
```c
//C Code
// 分配scratchpad内存
int8_t *a = gemm_malloc(16 * 16 * 4 * 1 * 196);  // [196, 1, 4, 16, 16]dims
int8_t *b = gemm_malloc(16 * 16 * 4 * 4 * 1 * 4); // [4, 1, 4, 4, 16, 16]dims
int32_t *res = gemm_acc_malloc(16 * 16 * 4 * 4);  // [4, 4, 16, 16]dims

```

------------------------------------
------------------------------------

- Core Operations

```py
#Python Code
do_zero_acc_i32(...)           # Initialize the accumulator to zero
do_ld_i8_block_id1(...)       # Load a block of Matrix A
do_ld_i8_block_id2(...)       # Load a block of Matrix B
do_matmul_acc_i8(...)         # Perform matrix multiplication and accumulation
do_st_acc_i8(...)             # Store the result
```
```c
//C Code
gemmini_extended_mvin(...)      // Move data into the zero register
gemmini_extended_mvin2(...)     // Load Matrix A into the scratchpad
gemmini_extended_mvin3(...)     // Load Matrix B data
gemmini_extended_preload(...)   // Preload data into the compute unit
gemmini_extended_compute_preloaded(...) // Execute matrix multiplication
gemmini_extended_mvout(...)     // Output the result
```
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
### Auto generate access stride
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

### the optimization of EXO-matmul
In EXO, optimization consists of two steps:
1. General Optimization and
2. Targeted Optimization.

Let's analyze general optimization first:
```c
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

    # Tile outer loops
    # gemmini = tile_outer_loops(gemmini)
    # print("\n=== Stage 3: After tiling outer loops ===")
    # print(gemmini)
    gemmini = divide_loop(gemmini, "i", 16, ["i", "i_in"], perfect=True)
    print("\n=== Stage 3.1: After divide loop i ===")
    print(gemmini)
    gemmini = old_reorder(gemmini, "i_in j")
    print("\n=== Stage 3.2: After reorder loop i_in and j ===")
    print(gemmini)
    gemmini = divide_loop(gemmini, "j", 64, ["j", "j_in"], perfect=True)
    print("\n=== Stage 3.3: After divide loop j ===")
    print(gemmini)
    gemmini = divide_loop(gemmini, "j_in", 16, ["j_in_o", "j_in_i"], perfect=True)
    gemmini = old_reorder(gemmini, "j_in_o j_in_i")
    print("\n=== Stage 3.4: After devide and reorder loop j_in ===")
    print(gemmini)

    # Lift res allocations
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=1, mode="col", size=16)
    print("\n=== Stage 4: After lifting res allocations ===")
    print(gemmini)
    """
    1. just declare a tensor res once instead of
    declaring a scalar res in very loop
    2. make the res be a continuous memory
    """

    # Fission outer blocks
    # gemmini = fission_outer_blocks(gemmini)
    # print("\n=== Stage 5: After fission outer blocks ===")
    # print(gemmini)
    gemmini = old_fission_after(gemmini, "res[_] = 0.0 #0", n_lifts=3)
    print("\n=== Stage 5.1: After fission after res[_] = 0.0 #0 ===")
    print(gemmini)
    gemmini = old_fission_after(gemmini, "for k in _:_ #0", n_lifts=3)
    print("\n=== Stage 5.2: After fission after for k in _:_ #0 ===")
    print(gemmini)
    gemmini = old_reorder(gemmini, "j_in_i j_in_o")
    gemmini = old_reorder(gemmini, "i_in k")
    gemmini = old_reorder(gemmini, "j_in_i k")
    gemmini = old_reorder(gemmini, "j_in_o k")
    # Here, 'k' is moved to the outermost loop.
    # Gemini's computations are parallel, allowing multiple data loads and parallel computations simultaneously.
    print("\n=== Stage 5.3: After reorder loop j_in_i, j_in_o, i_in, k ===")
    print(gemmini)

    # Fission inner blocks
    # gemmini = fission_inner_blocks(gemmini)
    # print("\n=== Stage 6: After fission inner blocks ===")
    # print(gemmini)
    gemmini = divide_loop(gemmini, "k", 64, ["ko", "k"], perfect=True)
    gemmini = divide_loop(gemmini, "k", 16, ["k", "ki"], perfect=True)
    print("\n=== Stage 6.1: After divide loop k ===")
    print(gemmini)
    gemmini = old_lift_alloc(gemmini, "a : i8", n_lifts=3)
    gemmini = old_lift_alloc(gemmini, "a : _ #0", n_lifts=1, mode="col")
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    print("\n=== Stage 6.2: After lift alloc a ===")
    print(gemmini)
    gemmini = old_reorder(gemmini, "ki j_in_o")
    gemmini = old_reorder(gemmini, "ki j_in_i")
    gemmini = old_lift_alloc(gemmini, "b : i8", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : i8", n_lifts=1, mode="col")
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=3)
    print("\n=== Stage 6.3: After lift alloc b ===")
    print(gemmini)
    gemmini = old_fission_after(gemmini, "a[_] = _", n_lifts=5)
    gemmini = old_fission_after(gemmini, "b[_] = _", n_lifts=5)
    print("\n=== Stage 6.4: After fission after a[_] = _ and b[_] = _ ===")
    print(gemmini)
    gemmini = old_reorder(gemmini, "j_in_i i_in")
    gemmini = old_reorder(gemmini, "ki i_in")
    gemmini = old_reorder(gemmini, "k i_in")
    gemmini = old_reorder(gemmini, "j_in_i ki")
    gemmini = old_reorder(gemmini, "j_in_o ki")
    gemmini = old_reorder(gemmini, "j_in_i i_in")
    print("\n=== Stage 6.5: After reorder loop j_in_i, ki, i_in ===")
    print(gemmini)

    # Replace with gemmini calls
    gemmini = replace_gemmini_calls(gemmini)
    print("\n=== Stage 7: After replacing with gemmini calls ===")
    print(gemmini)

    # Inline and lift config
    gemmini = inline_lift_config(gemmini)
    print("\n=== Stage 8: Final state after inline and lift config ===")
    print(gemmini)

    return cpu, gemmini

```

We've divided the analysis based on different code sections to illustrate the compilation optimization process. Overall, optimizing matrix multiplication (matmul) involves three main operations:
1. Loop Divide
    Tiling a specific dimension (typically to improve cache/SRAM locality).
    Facilitating hardware mapping (e.g., systolic arrays, tensor cores).
    Enabling parallel scheduling and vectorization.

   ```c
   Each time (k ∈ K_tile)：
   A_subtile[:, k] × B_subtile[k, :]  →  C_tile[:, :]

        A_subtile = (16×1)column vecto
        B_subtile = (1×16) row vector
     => Outer product: 16×16 tile 16x16 tile accumulated into res[...]
   ```
2. Loop Reorder
    Improves data access locality in inner loops.
    Optimizes computation order to align with data layout.
    Exposes parallelism.


3. Loop Fission
   Allows independent optimization of different parts.
   Facilitates scheduling and data reuse analysis.

   ```c
    # Original Code Example:
        for j_in_i:
        for j_in_o:
            for i_in:
            res[...] = 0.0
            for k:
                res[...] += ...
        #After Fission (split into two stages):
            Initialization stage
            Computation and write-back stage

   ```

### Process Illustration

Stage 3.1: Splitting 'i': This divides matrix A into 748 * (16x64) blocks, avoiding loading 12,544 rows at once. This reduces bandwidth pressure and cache miss rates.
![](/docs/Images/exo-matmul_3.1.png)
Stage 3.2: Reordering: Changes from row-major to column-major. This initially reduces the cache-hit-rate but prepares for subsequent parallelization (tiling 'j').
![](/docs/Images/exo-matmul_3.2.png)
Stage 3.3: Splitting 'j': This divides matrix B into 4 * (64x64) blocks, leading to stronger cache locality and more continuous memory access. It also facilitates mapping tiles to parallel computing units in hardware.
![](/docs/Images/exo-matmul_3.3.png)
Stage 3.4: Further Splitting 'j_in': This further divides matrix B into 4 * (4 * (64x16)) blocks, allowing for finer-grained access: B[k,64∗j+(16∗j_in_o+j_in_i)]. This is beneficial for 16-lane SIMD or Sysloic.
![](/docs/Images/exo-matmul_3.4.png)
Stage 5.1: Loop Fission: This affects scheduling and resource utilization. It can avoid redundant initialization and, after splitting, facilitate further unrolling of other parts.
Stage 6.1: Dividing 'k': The divide operation on 'k' transforms the original 64 'k' iterations into a triple loop: k_o − k − k_i
​This is equivalent to tiling the 'k' dimension, performing a small-scale outer product each time.
