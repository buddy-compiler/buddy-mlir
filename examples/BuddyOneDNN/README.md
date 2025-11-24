# BuddyOneDNN: External Operator Library Integration

This example demonstrates how to integrate external operator libraries (oneDNN) into the buddy-mlir framework by replacing specific operators (e.g., MatMul) with optimized external library calls at the Buddy Graph level.

## Quick Start

### Prerequisites

- **oneDNN:** 3.9.1+ (install via conda: `conda install -c conda-forge onednn`)
- **buddy-mlir:** Built from source
- **PyTorch:** 2.0+

### Build and Run

```bash
# In buddy-mlir root directory
cd build

# Build (automatically executes all compilation stages)
cmake --build . --target buddy-onednn-run

# Run
./bin/buddy-onednn-run
```

Expected output:
```
Result[0][0] = 16.5
Result[0][1] = 16.5
...
All values are 16.5 (matmul: 16.0, add bias: +0.5, relu: 16.5)
```

## What's Modified

### Frontend Changes

1. **`frontend/Python/graph/transform/onednn_replace.py`**
   - Transform pass to replace MatmulOp with CallOp

2. **`frontend/Python/graph/graph.py`**
   - Auto-generate external function declarations with `llvm.emit_c_interface` attribute

3. **`frontend/Python/ops/func.py`**
   - Support TOSA dialect's tensor types in CallOp lowering

4. **`frontend/Python/ops/tosa.py`**
   - Merge func.ops_registry to support CallOp

5. **`frontend/Python/graph/transform/__init__.py`**
   - Export transform functions

### Example Files

- **`onednn_ops.h/cpp`**: oneDNN C wrapper
- **`import_simple_model.py`**: PyTorch model import script
- **`CMakeLists.txt`**: Complete build configuration

## Compilation Flow

### Traditional (TOSA/Linalg)
```
PyTorch → Buddy Graph (MatmulOp) → TOSA → Linalg → Optimized Loops → Machine Code
```

### With External Library
```
PyTorch → Buddy Graph (MatmulOp)
  → [Transform: replace_matmul_with_onednn]
  → Buddy Graph (CallOp)
  → MLIR (func.call @onednn_matmul_f32)
  → Executable (linked with libonednn_ops.so)
```

## Key Technical Points

1. **Transform Order**: Must apply `replace_matmul_with_onednn` BEFORE `simply_fuse`
2. **OpType Restoration**: Restore `call_op._op_type = OpType.Unfusable` after `displace_node()`
3. **Tensor Types**: TOSA uses `RankedTensorType`, not `MemRefType`
4. **C ABI**: External functions require `llvm.emit_c_interface` attribute
5. **CMake Integration**: Fully integrated into buddy-mlir build system

## Build Stages

CMakeLists.txt implements 6 build stages:

1. Compile oneDNN wrapper library (`libonednn_ops.so`)
2. Generate MLIR from PyTorch (run Python script)
3. MLIR optimization (TOSA → Linalg)
4. MLIR lowering (Linalg → LLVM)
5. Code generation (LLVM IR → object file)
6. Link executable

## Output Files

Generated in `build/examples/BuddyOneDNN/`:

- `libonednn_ops.so` - oneDNN wrapper library
- `subgraph0.mlir` - Original MLIR (TOSA dialect)
- `subgraph0-opt.mlir` - Optimized MLIR (Linalg dialect)
- `subgraph0-llvm.mlir` - LLVM dialect MLIR
- `subgraph0.o` - Object file
- `buddy-onednn-run` - Executable (in `build/bin/`)

## Extending to Other Libraries

This pattern can be extended to other external operator libraries (MKL, cuBLAS, etc.) by:

1. Creating a transform pass (e.g., `replace_conv_with_mkl.py`)
2. Implementing a C wrapper for the library
3. Adding CMake build configuration

## References

- [oneDNN Documentation](https://oneapi-src.github.io/oneDNN/)
- [MLIR TOSA Dialect](https://mlir.llvm.org/docs/Dialects/TOSA/)
- [buddy-mlir Project](https://github.com/buddy-compiler/buddy-mlir)


## License

Apache 2.0
