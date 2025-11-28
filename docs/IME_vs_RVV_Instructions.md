# IME vs RVV 指令对比

## 快速理解

```
┌─────────────────────────────────────────┐
│         完整的IME程序                    │
├─────────────────────────────────────────┤
│  1. vsetvli  (RVV配置)                  │
│  2. vxor.vv  (RVV初始化)                │
│  3. vle8.v   (RVV加载A)                 │
│  4. vle8.v   (RVV加载B)                 │
│  5. vmadot   (IME计算) ← 唯一的IME指令  │
│  6. vse32.v  (RVV存储)                  │
└─────────────────────────────────────────┘
```

---

## 指令分类详解

### 1. RVV指令（RISC-V Vector Extension）

RVV是RISC-V的标准向量扩展，提供通用向量操作。

#### 配置指令
| 指令 | 功能 | 示例 |
|------|------|------|
| `vsetvli` | 设置向量长度和类型 | `vsetvli t0, zero, e8, m1` |

#### 数据移动指令
| 指令 | 功能 | 示例 |
|------|------|------|
| `vle8.v` | 加载int8向量 | `vle8.v v0, (a0)` |
| `vle16.v` | 加载int16向量 | `vle16.v v0, (a0)` |
| `vle32.v` | 加载int32向量 | `vle32.v v0, (a0)` |
| `vse8.v` | 存储int8向量 | `vse8.v v0, (a0)` |
| `vse16.v` | 存储int16向量 | `vse16.v v0, (a0)` |
| `vse32.v` | 存储int32向量 | `vse32.v v0, (a0)` |

#### 算术指令
| 指令 | 功能 | 示例 |
|------|------|------|
| `vadd.vv` | 向量加法 | `vadd.vv v0, v1, v2` |
| `vmul.vv` | 向量乘法 | `vmul.vv v0, v1, v2` |
| `vxor.vv` | 向量异或（常用于清零） | `vxor.vv v0, v0, v0` |

---

### 2. IME指令（SpacemiT扩展）

IME是SpacemiT对RVV的扩展，**只添加矩阵乘法指令**。

#### 矩阵乘法指令（全部5条）
| 指令 | 功能 | 操作数类型 | 结果类型 |
|------|------|-----------|---------|
| `vmadot` | 矩阵乘法累加 | int8 × int8 | int32 |
| `vmadotu` | 矩阵乘法累加 | uint8 × uint8 | int32 |
| `vmadotsu` | 矩阵乘法累加 | int8 × uint8 | int32 |
| `vmadotus` | 矩阵乘法累加 | uint8 × int8 | int32 |
| `vfmadot` | 矩阵乘法累加 | fp16 × fp16 | fp16 |

**注意**：
- IME **没有**加载/存储指令
- IME **没有**配置指令
- IME **只有**计算指令

---

## 完整代码对比

### 汇编代码
```assembly
# ========================================
# 完整的矩阵乘法程序
# ========================================

# 1. 配置向量环境 (RVV)
vsetvli  t0, zero, e32, m2      # 设置int32, LMUL=2
vxor.vv  v28, v28, v28          # 清零累加器

# 2. 加载数据 (RVV)
vsetvli  t0, zero, e8, m1       # 切换到int8
vle8.v   v0, (a0)               # 加载矩阵A
vle8.v   v1, (a1)               # 加载矩阵B

# 3. 计算 (IME) ← 唯一的IME指令！
vmadot   v28, v0, v1            # C += A × B

# 4. 存储结果 (RVV)
vsetvli  t0, zero, e32, m2      # 切换回int32
vse32.v  v28, (a2)              # 存储矩阵C
```

### MLIR代码
```mlir
func.func @matmul(%a: memref<4x8xi8>, %b: memref<8x4xi8>) -> memref<4x4xi32> {
  %c = memref.alloc() : memref<4x4xi32>
  
  // 初始化 (标准MLIR操作，会lowering到RVV)
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
  
  // 矩阵乘法 (IME操作)
  ime.vmadot %c, %a, %b : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  
  return %c : memref<4x4xi32>
}
```

---

## 指令来源总结表

| 功能类别 | 具体操作 | 指令来源 | 在MLIR中如何表示 |
|---------|---------|---------|----------------|
| **配置** | 设置向量长度 | RVV | 自动生成（不需要显式操作） |
| **初始化** | 清零累加器 | RVV | `linalg.fill` 或 `arith.constant` |
| **加载** | 从内存加载到寄存器 | RVV | `memref.load` 或 `vector.load` |
| **计算** | 矩阵乘法累加 | **IME** | **`ime.vmadot`** |
| **存储** | 从寄存器存储到内存 | RVV | `memref.store` 或 `vector.store` |

---

## IME Dialect需要实现什么？

### ✅ 需要实现（5个操作）
```mlir
ime.vmadot   %c, %a, %b : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
ime.vmadotu  %c, %a, %b : memref<4x4xi32>, memref<4x8xui8>, memref<8x4xui8>
ime.vmadotsu %c, %a, %b : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xui8>
ime.vmadotus %c, %a, %b : memref<4x4xi32>, memref<4x8xui8>, memref<8x4xi8>
ime.vfmadot  %c, %a, %b : memref<4x4xf16>, memref<4x8xf16>, memref<8x4xf16>
```

### ❌ 不需要实现
- ❌ `ime.load` - 使用标准的`memref.load`
- ❌ `ime.store` - 使用标准的`memref.store`
- ❌ `ime.config` - 不需要配置操作
- ❌ `ime.init` - 使用标准的`linalg.fill`

---

## 为什么这样设计？

### 优点
1. **简洁性**：IME只关注核心功能（矩阵计算）
2. **兼容性**：完全兼容RVV生态
3. **低成本**：复用RVV的寄存器和基础设施
4. **易集成**：与现有RVV代码无缝集成

### 对比其他加速器

| 特性 | Gemmini | IME |
|------|---------|-----|
| 数据移动 | 专用指令（mvin/mvout） | 复用RVV |
| 配置 | 专用指令（config_*） | 复用RVV |
| 计算 | 专用指令 | 专用指令 |
| 寄存器 | Scratchpad | 向量寄存器 |
| 复杂度 | 高 | 低 |

---

## 实际使用示例

### 场景：计算两个4×8和8×4矩阵的乘法

#### 步骤1：准备数据（C代码）
```c
int8_t A[32];  // 4×8
int8_t B[32];  // 8×4
int32_t C[16]; // 4×4
```

#### 步骤2：编写汇编（混合RVV和IME）
```assembly
# 使用RVV配置
vsetvli t0, zero, e32, m2
vxor.vv v28, v28, v28

# 使用RVV加载
vsetvli t0, zero, e8, m1
vle8.v  v0, (A_addr)
vle8.v  v1, (B_addr)

# 使用IME计算 ← 关键！
vmadot  v28, v0, v1

# 使用RVV存储
vsetvli t0, zero, e32, m2
vse32.v v28, (C_addr)
```

#### 步骤3：编写MLIR
```mlir
func.func @matmul() {
  %a = memref.get_global @matA : memref<4x8xi8>
  %b = memref.get_global @matB : memref<8x4xi8>
  %c = memref.alloc() : memref<4x4xi32>
  
  // 标准操作（会lowering到RVV）
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
  
  // IME操作
  ime.vmadot %c, %a, %b : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  
  return
}
```

---

## 总结

### 核心要点
1. **IME = 5条计算指令**
2. **数据移动 = RVV指令**
3. **IME Dialect只需实现5个操作**
4. **其他功能使用标准MLIR操作**

### 记忆口诀
```
配置用RVV，加载用RVV
计算用IME，存储用RVV
IME只管算，其他全靠RVV
```
