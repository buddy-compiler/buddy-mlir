# Tile Dialect

The Tile Dialect automatically tiles arbitrarily sized matrix operations into fixed-size tiles that the hardware can process, emitting loops plus Buckyball Dialect calls.

## Core Operations

### `tile.tile_matmul`
```mlir
tile.tile_matmul %A %B %C : memref<MxK xi8> memref<KxN xi8> memref<MxN xi32>
```

**Inputs**: memrefs of arbitrary size (M, K, N need not be multiples of 16)
**Output**: none (in-place write to %C)
**Semantics**: C = A × B, automatically tiled into tiles aligned to bank width (16) and integer multiples of bank depth

**Lowering strategy** (`-convert-tile-to-buckyball`):
1. **Padding**: if K or N is not a multiple of 16, allocate a padded buffer and zero-fill
2. **Tiling**: split M×K×N into `mTile × kTile × nTile` tiles, each satisfying:
   - K and N are multiples of 16 (bank width alignment)
   - `mTile * kTile + kTile * nTile ≤ bankDepth` (bank depth constraint)
   - `mTile * (nTile/16) ≤ 256` (accumulator mvout depth limit)
   - `mTile * (kTile/16) ≤ 1024` and `kTile * (nTile/16) ≤ 1024` (i8 mvin depth limits)
3. **Loop generation**: emit three nested `scf.for` loops over all tiles
4. **Subview + Buckyball**: slice each tile with `memref.subview`, then call `buckyball.matmul`

### `tile.tile_transpose`
Matrix transpose, automatically tiled into bank-aligned tiles.

### `tile.tile_conv2d`
2D convolution (NHWC layout); lowering expands to an Im2col + MatMul sequence.

## Design Notes

- **Tile ≠ Bank**: a tile is a memref slice; a bank is a physical scratchpad unit on the hardware
- **Compiler responsibility**: the Tile Dialect decomposes arbitrary-sized problems into subproblems within hardware constraints
- **Hardware-agnostic**: tile ops do not depend on the physical bank count (16); they only depend on bank width (16) and depth (1024)
