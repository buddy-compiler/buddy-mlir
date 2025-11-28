# SpacemiT IME 指令集参考文档

## 概述

SpacemiT IME (Intelligent Matrix Extension) 是一个RISC-V矩阵扩展，提供高效的矩阵乘法累加指令。

### 核心特性

- **复用向量寄存器**：基于RISC-V Vector Extension (RVV)，复用向量寄存器
- **灵活的向量长度**：支持VLEN从128位到4096位
- **丰富的数据类型**：支持整数（int4/8/16）和浮点（fp4/8/16/bfp16）
- **二进制可移植**：几乎在所有支持的VLEN配置下可移植

### 矩阵维度

所有IME指令执行以下矩阵运算：
```
A (M×K) × B (K×N) → C (M×N)
```

其中：
- **M, N, K**：矩阵维度，取决于VLEN和SEW配置
- **Copies**：副本数量，可以是1或2
  - `Copies = (sqrt(VLEN/64) == floor(sqrt(VLEN/64)) ? 1 : 2)`

---

## 指令分类

IME指令分为两大类：
1. **基本矩阵乘法累加指令**（无滑动窗口）
2. **滑动窗口矩阵乘法累加指令**（用于卷积等操作）

每类又分为：
- **整数指令**（OPMMA类别）
- **浮点指令**（OPFMMA类别）

---

## 一、基本矩阵乘法累加指令

### 1.1 整数矩阵乘法累加指令

#### 指令列表

| 指令 | 操作数A类型 | 操作数B类型 | 累加器C类型 | 说明 |
|------|------------|------------|------------|------|
| `vmadot` | int4/int8/int16 | int4/int8/int16 | int32 | signed × signed |
| `vmadotu` | uint4/uint8/uint16 | uint4/uint8/uint16 | int32 | unsigned × unsigned |
| `vmadotsu` | int4/int8/int16 | uint4/uint8/uint16 | int32 | signed × unsigned |
| `vmadotus` | uint4/uint8/uint16 | int4/int8/int16 | int32 | unsigned × signed |

#### 指令格式

```assembly
vmadot   vd, vs1, vs2    # vd(C) += vs1(A) × vs2(B)
vmadotu  vd, vs1, vs2
vmadotsu vd, vs1, vs2
vmadotus vd, vs1, vs2
```

#### 寄存器约束

- **vd**：目标寄存器（输出C矩阵），**索引必须为偶数**
- **vs1**：源寄存器1（输入A矩阵）
- **vs2**：源寄存器2（输入B矩阵）
- **输出**：结果存储在两个连续寄存器中（vd和vd+1）

#### 操作语义

```c
Copies = (sqrt(VLEN/64) == floor(sqrt(VLEN/64)) ? 1 : 2)
for (cp = 0; cp < Copies; cp++) {
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < K; k++) {
                C[cp * M * N + i * K + j] += 
                    int32(A[cp * M * K + i * K + k] * B[cp * K * N + j * K + k]);
            }
        }
    }
}
```

### 1.2 浮点矩阵乘法累加指令

#### 指令列表

| 指令 | 操作数A类型 | 操作数B类型 | 累加器C类型 | 说明 |
|------|------------|------------|------------|------|
| `vfmadot` | fp4/fp8/fp16/bfp16 | fp4/fp8/fp16/bfp16 | fp16/bfp16 | 浮点矩阵乘法 |

#### 指令格式

```assembly
vfmadot vd, vs1, vs2    # vd(C) += vs1(A) × vs2(B)
```

#### 寄存器约束

- **vd**：目标寄存器（输出C矩阵）
- **vs1**：源寄存器1（输入A矩阵）
- **vs2**：源寄存器2（输入B矩阵）
- **输出**：结果存储在单个寄存器中（与整数指令不同）

#### 操作语义

```c
// FP = fp16 or bfp16
Copies = (sqrt(VLEN/64) == floor(sqrt(VLEN/64)) ? 1 : 2)
for (cp = 0; cp < Copies; cp++) {
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < K; k++) {
                C[cp * M * N + i * K + j] += 
                    FP(A[cp * M * K + i * K + k] * B[cp * K * N + j * K + k]);
            }
        }
    }
}
```

---

## 二、滑动窗口矩阵乘法累加指令

滑动窗口指令从两个连续寄存器（vs1和vs1+1）中选择指定的值作为矩阵A，用于卷积等操作。

### 2.1 整数滑动窗口指令

#### 指令列表（按滑动值分类）

**固定滑动值 = 1**
| 指令 | 操作数A类型 | 操作数B类型 | 累加器C类型 |
|------|------------|------------|------------|
| `vmadot1` | int4/int8/int16 | int4/int8/int16 | int32 |
| `vmadot1u` | uint4/uint8/uint16 | uint4/uint8/uint16 | int32 |
| `vmadot1su` | int4/int8/int16 | uint4/uint8/uint16 | int32 |
| `vmadot1us` | uint4/uint8/uint16 | int4/int8/int16 | int32 |

**固定滑动值 = 2**
| 指令 | 操作数A类型 | 操作数B类型 | 累加器C类型 |
|------|------------|------------|------------|
| `vmadot2` | int4/int8/int16 | int4/int8/int16 | int32 |
| `vmadot2u` | uint4/uint8/uint16 | uint4/uint8/uint16 | int32 |
| `vmadot2su` | int4/int8/int16 | uint4/uint8/uint16 | int32 |
| `vmadot2us` | uint4/uint8/uint16 | int4/int8/int16 | int32 |

**固定滑动值 = 3**
| 指令 | 操作数A类型 | 操作数B类型 | 累加器C类型 |
|------|------------|------------|------------|
| `vmadot3` | int4/int8/int16 | int4/int8/int16 | int32 |
| `vmadot3u` | uint4/uint8/uint16 | uint4/uint8/uint16 | int32 |
| `vmadot3su` | int4/int8/int16 | uint4/uint8/uint16 | int32 |
| `vmadot3us` | uint4/uint8/uint16 | int4/int8/int16 | int32 |

**可变滑动值（通过寄存器t0指定）**
| 指令 | 操作数A类型 | 操作数B类型 | 累加器C类型 |
|------|------------|------------|------------|
| `vmadotn` | int4/int8/int16 | int4/int8/int16 | int32 |
| `vmadotnu` | uint4/uint8/uint16 | uint4/uint8/uint16 | int32 |
| `vmadotnsu` | int4/int8/int16 | uint4/uint8/uint16 | int32 |
| `vmadotnus` | uint4/uint8/uint16 | int4/int8/int16 | int32 |

#### 指令格式

```assembly
# 固定滑动值
vmadot1   vd, vs1, vs2
vmadot1u  vd, vs1, vs2
vmadot1su vd, vs1, vs2
vmadot1us vd, vs1, vs2

vmadot2   vd, vs1, vs2
vmadot2u  vd, vs1, vs2
vmadot2su vd, vs1, vs2
vmadot2us vd, vs1, vs2

vmadot3   vd, vs1, vs2
vmadot3u  vd, vs1, vs2
vmadot3su vd, vs1, vs2
vmadot3us vd, vs1, vs2

# 可变滑动值（通过t0寄存器）
vmadotn   vd, vs1, vs2, t0
vmadotnu  vd, vs1, vs2, t0
vmadotnsu vd, vs1, vs2, t0
vmadotnus vd, vs1, vs2, t0
```

#### 寄存器约束

- **vd**：目标寄存器（输出C矩阵），**索引必须为偶数**
- **vs1**：源寄存器1（输入A矩阵），**索引必须为偶数**（需要vs1和vs1+1）
- **vs2**：源寄存器2（输入B矩阵）
- **t0**：滑动值（仅用于vmadotn系列指令）
- **输出**：结果存储在两个连续寄存器中（vd和vd+1）

#### 操作语义

```c
Copies = (sqrt(VLEN/64) == floor(sqrt(VLEN/64)) ? 1 : 2)
for (cp = 0; cp < Copies; cp++) {
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < K; k++) {
                C[cp * M * N + i * K + j] +=
                    int32(A[cp * M * K + slide * K + i * K + k] * B[cp * K * N + j * K + k]);
            }
        }
    }
}
```

### 2.2 浮点滑动窗口指令

#### 指令列表

| 滑动值 | 指令 | 操作数A类型 | 操作数B类型 | 累加器C类型 |
|--------|------|------------|------------|------------|
| 1 | `vfmadot1` | fp4/fp8/fp16/bfp16 | fp4/fp8/fp16/bfp16 | fp16/bfp16 |
| 2 | `vfmadot2` | fp4/fp8/fp16/bfp16 | fp4/fp8/fp16/bfp16 | fp16/bfp16 |
| 3 | `vfmadot3` | fp4/fp8/fp16/bfp16 | fp4/fp8/fp16/bfp16 | fp16/bfp16 |
| n | `vfmadotn` | fp4/fp8/fp16/bfp16 | fp4/fp8/fp16/bfp16 | fp16/bfp16 |

#### 指令格式

```assembly
vfmadot1 vd, vs1, vs2
vfmadot2 vd, vs1, vs2
vfmadot3 vd, vs1, vs2
vfmadotn vd, vs1, vs2, t0    # 可变滑动值
```

#### 寄存器约束

- **vd**：目标寄存器（输出C矩阵）
- **vs1**：源寄存器1（输入A矩阵），**索引必须为偶数**
- **vs2**：源寄存器2（输入B矩阵）
- **t0**：滑动值（仅用于vfmadotn）
- **输出**：结果存储在单个寄存器中

---

## 三、指令总结

### 完整指令列表（共24条）

#### 基本指令（5条）
1. `vmadot` - 整数矩阵乘法（signed × signed）
2. `vmadotu` - 整数矩阵乘法（unsigned × unsigned）
3. `vmadotsu` - 整数矩阵乘法（signed × unsigned）
4. `vmadotus` - 整数矩阵乘法（unsigned × signed）
5. `vfmadot` - 浮点矩阵乘法

#### 滑动窗口指令（19条）

**整数滑动窗口（16条）**
- slide=1: `vmadot1`, `vmadot1u`, `vmadot1su`, `vmadot1us`
- slide=2: `vmadot2`, `vmadot2u`, `vmadot2su`, `vmadot2us`
- slide=3: `vmadot3`, `vmadot3u`, `vmadot3su`, `vmadot3us`
- slide=n: `vmadotn`, `vmadotnu`, `vmadotnsu`, `vmadotnus`

**浮点滑动窗口（4条）**
- `vfmadot1`, `vfmadot2`, `vfmadot3`, `vfmadotn`

---

## 四、配置示例

### VLEN=256, SEW=8 配置

- **M = 4, K = 8, N = 4**
- **Copies = 1**
- **矩阵A**: 4×8 (32个int8元素)
- **矩阵B**: 8×4 (32个int8元素)
- **矩阵C**: 4×4 (16个int32元素，需要2个寄存器)

### VLEN=512, SEW=8 配置

- **M = 4, K = 8, N = 4**
- **Copies = 2**
- **矩阵A**: 2×(4×8) (64个int8元素)
- **矩阵B**: 2×(8×4) (64个int8元素)
- **矩阵C**: 2×(4×4) (32个int32元素，需要2个寄存器)

---

## 五、关键特性对比

| 特性 | 整数指令 | 浮点指令 |
|------|---------|---------|
| 输出寄存器数量 | 2个（vd, vd+1） | 1个（vd） |
| vd索引约束 | 必须为偶数 | 无约束 |
| 累加器类型 | int32 | fp16/bfp16 |
| 数据类型 | int4/8/16, uint4/8/16 | fp4/8/16, bfp16 |
| 符号变体 | 4种（ss, uu, su, us） | 1种 |

---

## 六、使用注意事项

1. **寄存器索引约束**
   - 整数指令的vd必须为偶数
   - 滑动窗口指令的vs1必须为偶数

2. **数据布局**
   - 矩阵数据在向量寄存器中按特定布局存储
   - 矩阵B可能需要预先打包（pack）以匹配硬件要求

3. **VLEN依赖**
   - 矩阵维度（M, N, K）取决于VLEN和SEW配置
   - Copies数量由VLEN决定

4. **累加语义**
   - 所有指令都是累加操作（+=），不是赋值操作
   - 使用前需要初始化目标寄存器为0

---

## 参考资料

- [SpacemiT IME Extension Specification](https://github.com/spacemit-com/riscv-ime-extension-spec)
- RISC-V Vector Extension Specification

