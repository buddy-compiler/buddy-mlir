# IME 指令使用示例详解

本文档详细讲解如何使用SpacemiT IME指令进行矩阵乘法运算。

## 示例概述

**示例代码**：`riscv-ime-extension-spec/example/vmadot-gemm-demo.c`

**功能**：使用`vmadot`指令计算4×8矩阵与8×4矩阵的乘法

**配置**：
- VLEN = 256位
- SEW = 8位（int8数据类型）
- M = 4, K = 8, N = 4
- Copies = 1

**矩阵运算**：
```
A (4×8) × B (8×4) → C (4×4)
```

---

## 代码结构分析

### 1. 参考实现（标准矩阵乘法）

```c
void Referece_Gemm(size_t M, size_t N, size_t K,
                   const int8_t* A, const int8_t* B, int32_t* C) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (size_t k = 0; k < K; ++k) {
                int8_t a = A[m * K + k];      // A的第m行第k列
                int8_t b = B[k * N + n];      // B的第k行第n列
                acc += a * b;
            }
            C[m * N + n] = acc;               // C的第m行第n列
        }
    }
}
```

**说明**：
- 标准的三重循环矩阵乘法
- 用于验证IME指令的正确性
- 时间复杂度：O(M×N×K) = O(4×4×8) = 128次乘法

---

### 2. 矩阵B的打包（Gemm_packB）

```c
void Gemm_packB(size_t ROW, size_t COL,
                const int8_t* B, int8_t* packedB) {
    __asm__ volatile (
        "addi         t6, zero, 8             \n\t"  // stride = 8
        "vsetvli      t0, zero, e8, mf8       \n\t"  // 设置向量长度
        
        "LOOP_ROW%=:                          \n\t"
        "addi         %[ROW], %[ROW], -1      \n\t"
        
        "LOOP_COL%=:                          \n\t"
        "vle8.v       v0, (%[SRC])            \n\t"  // 加载4个元素
        "addi         %[SRC], %[SRC], 4       \n\t"
        "vsse8.v      v0, (%[DST]), t6        \n\t"  // 以stride=8存储
        "addi         %[DST], %[DST], 1       \n\t"
        
        "bnez         %[ROW], LOOP_ROW%=      \n\t"
        
        : [SRC] "+r"(B), [DST] "+r"(packedB), [ROW] "+r"(ROW)
        : [COL] "r"(COL)
        : "cc", "t6", "t0");
}
```

**为什么需要打包？**

IME指令要求矩阵B的数据布局与标准行主序不同。打包操作将B矩阵重新排列以匹配硬件要求。

**原始B矩阵布局**（行主序，8×4）：
```
B[0,0] B[0,1] B[0,2] B[0,3]
B[1,0] B[1,1] B[1,2] B[1,3]
B[2,0] B[2,1] B[2,2] B[2,3]
...
B[7,0] B[7,1] B[7,2] B[7,3]
```

**打包后的B矩阵布局**（列主序，按K维度交错）：
```
B[0,0] B[1,0] B[2,0] B[3,0] B[4,0] B[5,0] B[6,0] B[7,0]
B[0,1] B[1,1] B[2,1] B[3,1] B[4,1] B[5,1] B[6,1] B[7,1]
B[0,2] B[1,2] B[2,2] B[3,2] B[4,2] B[5,2] B[6,2] B[7,2]
B[0,3] B[1,3] B[2,3] B[3,3] B[4,3] B[5,3] B[6,3] B[7,3]
```

**打包过程**：
1. 使用`vle8.v`加载一行的4个元素
2. 使用`vsse8.v`以stride=8存储，实现转置效果

---

### 3. IME矩阵乘法（Gemm_vmadot）

```c
void Gemm_vmadot(size_t M, size_t N, size_t K,
                 const int8_t* A, const int8_t* B, int32_t* C) {
    __asm__ volatile(
        // 初始化累加器为0
        "vsetvli      t0, zero, e32, m2       \n\t"  // 设置为int32类型
        "vxor.vv      v28, v28, v28           \n\t"  // v28 = 0 (累加器)
        
        // 设置为int8类型进行矩阵乘法
        "vsetvli      t0, zero, e8, m1        \n\t"
        "LOOP_K%=:                            \n\t"
        
        // 加载矩阵A (4×8 = 32 bytes)
        "vle8.v       v0, (%[A])              \n\t"
        "addi         %[A], %[A], 32          \n\t"
        
        // 加载矩阵B (打包后的8×4 = 32 bytes)
        "vle8.v       v1, (%[B])              \n\t"
        "addi         %[B], %[B], 32          \n\t"
        
        // 执行矩阵乘法累加
        "vmadot       v28, v0, v1             \n\t"  // v28 += v0 × v1
        
        // 存储结果 (4×4 = 16个int32 = 64 bytes)
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        
        : [A] "+r"(A), [B] "+r"(B), [C] "+r"(C), [M] "+r"(M)
        : [K] "r"(K), [N] "r"(N)
        : "cc");
}
```

**关键步骤解析**：

#### 步骤1：初始化累加器
```assembly
vsetvli t0, zero, e32, m2    # 设置向量类型为int32，LMUL=2
vxor.vv v28, v28, v28        # v28 = 0（清零累加器）
```
- 因为`vmadot`是累加操作（+=），所以必须先清零
- 使用`e32`因为输出是int32类型
- `m2`表示使用2个寄存器（v28和v29）存储16个int32结果

#### 步骤2：加载矩阵A
```assembly
vsetvli t0, zero, e8, m1     # 切换到int8类型
vle8.v  v0, (%[A])           # 加载32个int8元素到v0
```
- 矩阵A (4×8) = 32个int8元素
- 全部加载到向量寄存器v0中

#### 步骤3：加载矩阵B（打包后）
```assembly
vle8.v  v1, (%[B])           # 加载32个int8元素到v1
```
- 打包后的矩阵B也是32个int8元素
- 加载到向量寄存器v1中

#### 步骤4：执行矩阵乘法累加
```assembly
vmadot  v28, v0, v1          # v28 += v0 × v1
```

**这一条指令完成了什么？**

相当于执行以下伪代码：
```c
for (i = 0; i < 4; i++) {           // M=4
    for (j = 0; j < 4; j++) {       // N=4
        for (k = 0; k < 8; k++) {   // K=8
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

**单条指令完成128次乘法和累加！**

#### 步骤5：存储结果
```assembly
vsetvli t0, zero, e32, m2    # 切换回int32类型
vse32.v v28, (%[C])          # 存储16个int32结果
```
- 结果C (4×4) = 16个int32元素
- 存储在v28和v29两个寄存器中（共64字节）

---

## 寄存器使用情况

### 向量寄存器分配

| 寄存器 | 用途 | 数据类型 | 元素数量 |
|--------|------|---------|---------|
| v0 | 矩阵A | int8 | 32 (4×8) |
| v1 | 矩阵B（打包后） | int8 | 32 (8×4) |
| v28-v29 | 矩阵C（累加器） | int32 | 16 (4×4) |

### 标量寄存器

| 寄存器 | 用途 |
|--------|------|
| t0 | vsetvli返回值（向量长度） |
| t6 | stride值（打包时使用） |

---

## 性能对比

### 计算量
- **总乘法次数**：M×N×K = 4×4×8 = 128次
- **总加法次数**：M×N×K = 128次

### 指令数量对比

**标准实现**（Referece_Gemm）：
- 约 128×3 = 384条指令（每次乘加需要load、mul、add）

**IME实现**（Gemm_vmadot）：
- 初始化：2条
- 加载A：1条
- 加载B：1条
- 矩阵乘法：**1条** ← 关键！
- 存储C：1条
- **总计：6条指令**

**加速比**：约 64倍（指令数量）

---

## 完整执行流程

### 主函数执行步骤

```c
int main() {
    int8_t A[32];      // 4×8矩阵
    int8_t B[32];      // 8×4矩阵
    int8_t packB[32];  // 打包后的B矩阵
    int32_t C[16];     // 4×4结果矩阵
    int32_t CRef[16];  // 参考结果
    
    // 1. 随机初始化A和B
    for (size_t i = 0; i < 32; ++i) {
        A[i] = rand() % 256 - 128;     // [-128, 127]
        B[i] = rand() % 256 - 128;
    }
    
    // 2. 打包矩阵B
    Gemm_packB(8, 4, B, packB);
    
    // 3. 计算参考结果
    Referece_Gemm(4, 4, 8, A, B, CRef);
    
    // 4. 使用IME指令计算
    Gemm_vmadot(4, 4, 8, A, packB, C);
    
    // 5. 验证结果
    Test(4, 4, CRef, C);  // 断言CRef == C
    
    return 0;
}
```

---

## 关键要点总结

### 1. 数据准备
- ✅ 矩阵A：标准行主序布局，无需特殊处理
- ⚠️ 矩阵B：需要打包（转置+重排）以匹配硬件要求
- ✅ 矩阵C：初始化为0（因为vmadot是累加操作）

### 2. 指令使用
- `vsetvli`：配置向量长度和数据类型
- `vle8.v`：加载int8向量数据
- `vmadot`：**核心指令**，执行矩阵乘法累加
- `vse32.v`：存储int32结果

### 3. 寄存器约束
- vd（v28）索引必须为偶数（整数指令）
- 结果使用2个连续寄存器（v28, v29）

### 4. 性能优势
- 单条指令完成完整的矩阵乘法
- 大幅减少指令数量和内存访问
- 硬件加速的矩阵运算

---

## 编译和运行

```bash
# 编译
riscv64-unknown-linux-gnu-gcc -march=rv64gcv vmadot-gemm-demo.c -o gemm-vmadot-4x8x4

# 在QEMU上运行
qemu-riscv64 -cpu max,vlen=256 gemm-vmadot-4x8x4

# 在K1硬件上运行
./gemm-vmadot-4x8x4
```

**预期输出**：
```
Test successful. CRef equal to C.
*********************************
matrix A: 
  ...
matrix B: 
  ...
matrix C: 
  ...
*********************************
```
