# IME完整指令流程说明

## 核心概念

**IME只提供计算指令，数据加载/存储使用RVV标准指令**

---

## 指令分类

### 1. IME指令（计算）

IME扩展**只有5条基本计算指令**：

```assembly
vmadot   vd, vs1, vs2    # 整数矩阵乘法累加（signed × signed）
vmadotu  vd, vs1, vs2    # 整数矩阵乘法累加（unsigned × unsigned）
vmadotsu vd, vs1, vs2    # 整数矩阵乘法累加（signed × unsigned）
vmadotus vd, vs1, vs2    # 整数矩阵乘法累加（unsigned × signed）
vfmadot  vd, vs1, vs2    # 浮点矩阵乘法累加
```

**功能**：执行矩阵乘法累加运算
**输入**：向量寄存器vs1（矩阵A）、vs2（矩阵B）
**输出**：向量寄存器vd（矩阵C，累加）

---

### 2. RVV指令（数据移动和配置）

IME依赖RVV的以下指令：

#### 2.1 配置指令
```assembly
vsetvli  rd, rs1, vtypei    # 设置向量长度和类型
# 例如：
vsetvli  t0, zero, e8, m1   # SEW=8位, LMUL=1
vsetvli  t0, zero, e32, m2  # SEW=32位, LMUL=2
```

#### 2.2 加载指令
```assembly
vle8.v   vd, (rs1)          # 加载int8向量
vle16.v  vd, (rs1)          # 加载int16向量
vle32.v  vd, (rs1)          # 加载int32向量
vlef.v   vd, (rs1)          # 加载浮点向量
```

#### 2.3 存储指令
```assembly
vse8.v   vs3, (rs1)         # 存储int8向量
vse16.v  vs3, (rs1)         # 存储int16向量
vse32.v  vs3, (rs1)         # 存储int32向量
vsef.v   vs3, (rs1)         # 存储浮点向量
```

#### 2.4 初始化指令
```assembly
vxor.vv  vd, vs1, vs2       # 向量异或（用于清零）
vmv.v.i  vd, imm            # 向量赋立即数
```

---

## 完整使用流程

### 示例：4×8矩阵 × 8×4矩阵 = 4×4矩阵

```assembly
# ============================================
# 步骤1：配置向量环境（RVV指令）
# ============================================
vsetvli  t0, zero, e32, m2      # 设置为int32类型，LMUL=2
vxor.vv  v28, v28, v28          # 清零累加器v28（C矩阵）

# ============================================
# 步骤2：加载矩阵A（RVV指令）
# ============================================
vsetvli  t0, zero, e8, m1       # 切换到int8类型
vle8.v   v0, (a0)               # 从内存加载A矩阵到v0
                                # a0指向A矩阵的内存地址
                                # 加载32个int8元素（4×8）

# ============================================
# 步骤3：加载矩阵B（RVV指令）
# ============================================
vle8.v   v1, (a1)               # 从内存加载B矩阵到v1
                                # a1指向B矩阵的内存地址
                                # 加载32个int8元素（8×4，打包后）

# ============================================
# 步骤4：执行矩阵乘法（IME指令）
# ============================================
vmadot   v28, v0, v1            # v28 += v0 × v1
                                # 这是唯一的IME指令！
                                # 完成128次乘法和累加

# ============================================
# 步骤5：存储结果（RVV指令）
# ============================================
vsetvli  t0, zero, e32, m2      # 切换回int32类型
vse32.v  v28, (a2)              # 将v28存储到内存
                                # a2指向C矩阵的内存地址
                                # 存储16个int32元素（4×4）
```

---

## 寄存器使用

### 向量寄存器（v0-v31）
```
v0       : 矩阵A（输入）
v1       : 矩阵B（输入）
v28-v29  : 矩阵C（输出，累加器）
           注意：整数指令需要2个连续寄存器
```

### 标量寄存器
```
a0       : A矩阵的内存地址
a1       : B矩阵的内存地址
a2       : C矩阵的内存地址
t0       : vsetvli返回的向量长度
```

---

## C代码示例

```c
void ime_matmul(int8_t* A, int8_t* B, int32_t* C) {
    __asm__ volatile(
        // 1. 初始化累加器（RVV）
        "vsetvli  t0, zero, e32, m2       \n\t"
        "vxor.vv  v28, v28, v28           \n\t"
        
        // 2. 加载矩阵A（RVV）
        "vsetvli  t0, zero, e8, m1        \n\t"
        "vle8.v   v0, (%[A])              \n\t"
        
        // 3. 加载矩阵B（RVV）
        "vle8.v   v1, (%[B])              \n\t"
        
        // 4. 矩阵乘法（IME）
        "vmadot   v28, v0, v1             \n\t"
        
        // 5. 存储结果（RVV）
        "vsetvli  t0, zero, e32, m2       \n\t"
        "vse32.v  v28, (%[C])             \n\t"
        
        : // 输出操作数
        : [A] "r"(A), [B] "r"(B), [C] "r"(C)  // 输入操作数
        : "t0", "v0", "v1", "v28", "v29"      // 破坏的寄存器
    );
}
```

---

## 指令来源总结

| 功能 | 指令 | 来源 | 说明 |
|------|------|------|------|
| 配置向量 | `vsetvli` | RVV | 设置SEW和LMUL |
| 清零 | `vxor.vv` | RVV | 初始化累加器 |
| 加载数据 | `vle8.v`, `vle32.v` | RVV | 从内存加载到向量寄存器 |
| **矩阵乘法** | **`vmadot`** | **IME** | **核心计算指令** |
| 存储数据 | `vse8.v`, `vse32.v` | RVV | 从向量寄存器存储到内存 |

---

## 关键要点

### ✅ IME的定位
- IME是RVV的**扩展**，不是独立的指令集
- IME**复用**RVV的向量寄存器（v0-v31）
- IME**依赖**RVV的配置和数据移动指令

### ✅ 为什么这样设计？
1. **低成本**：复用RVV的寄存器和基础设施
2. **兼容性**：与RVV无缝集成
3. **简洁性**：只添加必要的计算指令

### ✅ 编程模型
```
1. 使用RVV配置向量环境
2. 使用RVV加载数据到向量寄存器
3. 使用IME执行矩阵计算
4. 使用RVV存储结果到内存
```

---

## MLIR Dialect设计影响

基于这个理解，IME Dialect应该：

### ✅ 需要实现
- `ime.vmadot` 等5条计算操作
- Lowering到LLVM intrinsics

### ❌ 不需要实现
- 数据加载/存储操作（使用标准的`memref.load`/`memref.store`）
- 向量配置操作（MLIR会自动处理）
- 寄存器分配（由LLVM backend处理）

### 示例MLIR代码
```mlir
func.func @matmul(%a: memref<4x8xi8>, %b: memref<8x4xi8>) -> memref<4x4xi32> {
  %c = memref.alloc() : memref<4x4xi32>
  
  // 初始化C为0（使用标准操作）
  %zero = arith.constant 0 : i32
  linalg.fill ins(%zero : i32) outs(%c : memref<4x4xi32>)
  
  // 矩阵乘法（使用IME操作）
  ime.vmadot %c, %a, %b : memref<4x4xi32>, memref<4x8xi8>, memref<8x4xi8>
  
  return %c : memref<4x4xi32>
}
```

Lowering后会生成：
```llvm
; 加载数据（标准LLVM IR）
%a_vec = load <32 x i8>, ptr %a_ptr
%b_vec = load <32 x i8>, ptr %b_ptr

; 矩阵乘法（IME intrinsic）
%c_vec = call <16 x i32> @llvm.riscv.ime.vmadot(<16 x i32> %c_vec, <32 x i8> %a_vec, <32 x i8> %b_vec)

; 存储结果（标准LLVM IR）
store <16 x i32> %c_vec, ptr %c_ptr
```

---

## 总结

**IME = 5条计算指令 + RVV的数据移动指令**

- 🔢 **计算**：使用IME的`vmadot`系列指令
- 📦 **数据移动**：使用RVV的`vle/vse`指令
- ⚙️ **配置**：使用RVV的`vsetvli`指令
- 🎯 **设计理念**：专注于矩阵计算，其他功能复用RVV
