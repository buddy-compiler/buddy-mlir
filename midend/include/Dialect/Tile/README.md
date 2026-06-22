# Tile Dialect

Tile Dialect 将任意尺寸的矩阵运算自动分块（tiling）为硬件可处理的固定尺寸 tile，生成循环 + Buckyball Dialect 调用。

## 核心操作

### `tile.tile_matmul`
```mlir
tile.tile_matmul %A %B %C : memref<MxK xi8> memref<KxN xi8> memref<MxN xi32>
```

**输入**：任意尺寸 memref（M、K、N 可以不是 16 的倍数）  
**输出**：无（in-place 写入 %C）  
**语义**：C = A × B，自动分块为 bank 宽度（16）× bank 深度整数倍的 tile

**Lowering 策略**（`-convert-tile-to-buckyball`）：
1. **Padding**：如果 K 或 N 不是 16 的倍数，分配 padded buffer 并 zero-fill
2. **Tiling**：将 M×K×N 分块为 `mTile × kTile × nTile`，每个 tile 满足：
   - K、N 是 16 的倍数（bank 宽度对齐）
   - `mTile * kTile + kTile * nTile ≤ bankDepth`（bank 深度约束）
   - `mTile * (nTile/16) ≤ 256`（accumulator mvout 深度限制）
   - `mTile * (kTile/16) ≤ 1024` 且 `kTile * (nTile/16) ≤ 1024`（i8 mvin 深度限制）
3. **循环生成**：生成 3 层 `scf.for` 循环遍历所有 tile
4. **Subview + Buckyball**：每个 tile 用 `memref.subview` 切片，调用 `buckyball.matmul`

### `tile.tile_transpose`
矩阵转置，自动分块为 bank 对齐的 tile。

### `tile.tile_conv2d`
2D 卷积（NHWC layout），lowering 为 Im2col + MatMul 序列。

## 设计要点

- **Tile ≠ Bank**：Tile 是 memref 的切片，Bank 是硬件 scratchpad 的物理单元
- **编译器职责**：Tile Dialect 负责将任意尺寸问题分解为硬件约束内的子问题
- **硬件无关**：Tile ops 不感知物理 bank 数量（16），只关心 bank 宽度（16）和深度（1024）

## Pass Pipeline

```
tile.tile_matmul
  ↓ -convert-tile-to-buckyball
scf.for + memref.subview + buckyball.matmul
  ↓ -lower-buckyball-to-bank-ssa
buckyball.bank_alloc + bank_mvin + bank_mul_warp16 + bank_mvout
  ↓ -assign-physical-banks
buckyball.mset + mvin + mul_warp16 + mvout
  ↓ -lower-buckyball
LLVM intrinsics (bb_mset / bb_mvin / bb_mul_warp16 / bb_mvout)
```
