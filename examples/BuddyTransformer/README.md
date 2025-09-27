# Buddy Compiler DeepSeek R1 Transformer Example

## Introduction
This example demonstrates how to use Buddy Compiler to compile a complete DeepSeek R1 1.5B transformer block to optimized MLIR code. The compilation pipeline includes frontend conversion from PyTorch to MLIR and staged optimization passes.

## Files Structure
- `transformer_model.py`: DeepSeek R1 transformer block PyTorch model definition
- `import-transformer.py`: Frontend script to convert PyTorch model to MLIR
- `transformer_runner.cpp`: C++ performance test runner
- `CMakeLists.txt`: Build configuration for compilation
- `README.md`: This documentation

## Model Configuration
- Hidden size: 1536
- Intermediate size: 8960
- Number of attention heads: 12
- Number of key-value heads: 2 (Grouped Query Attention)
- Head dimension: 128
- Maximum sequence length: 32768
- Default input: batch_size=1, seq_len=40

## How to Build

### Prerequisites
1. Build LLVM/MLIR and buddy-mlir following the main project instructions
2. Set up Python environment with required packages:
```bash
cd buddy-mlir
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export BUDDY_MLIR_BUILD_DIR=/path/to/buddy-mlir/build
export LLVM_MLIR_BUILD_DIR=/path/to/buddy-mlir/llvm/build
export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

### Build Commands

1. Enable the transformer example in CMake:
```bash
cd buddy-mlir/build
cmake -G Ninja .. -DBUDDY_TRANSFORMER_EXAMPLES=ON
```

2. Build the transformer example:

### One-step Compilation (Default/Production)

For complete build (fastest, no intermediate files):
```bash
ninja buddy-transformer
```

This generates `transformer-runner` executable directly.

### Staged Compilation (Manual debugging and analysis)

For frontend only (PyTorch to TOSA):
```bash
ninja buddy-transformer-frontend
```

For midend optimization (TOSA to optimized MLIR):
```bash
ninja buddy-transformer-midend
```

For backend lowering (optimized MLIR to LLVM MLIR):
```bash
ninja buddy-transformer-backend
```

For code generation (LLVM IR and object files):
```bash
ninja buddy-transformer-codegen
```

For staged executable:
```bash
ninja buddy-transformer-executable
```

## Running the Example

After building, run the performance test:
```bash
./bin/transformer-runner
```

## Generated Files

After building, the following files will be generated in the build directory:

### Frontend Output (PyTorch to MLIR)
- `graph.log`: Buddy Graph representation before lowering to TOSA
- `graph_fused.log`: Buddy Graph representation after fusion optimization
- `forward.mlir`: Main graph MLIR representation
- `subgraph0.mlir`: Subgraph MLIR representation
- `arg0.data`: Model parameters in binary format

### Midend Output (Optimized MLIR)
- `forward-midend.mlir`: Forward graph after midend optimization
- `subgraph0-midend.mlir`: Subgraph after midend optimization

### Backend Output (Lowered MLIR)
- `forward-backend.mlir`: Forward graph after backend lowering
- `subgraph0-backend.mlir`: Subgraph after backend lowering

### Code Generation Output
- `forward.ll`: Forward graph LLVM IR
- `subgraph0.ll`: Subgraph LLVM IR
- `forward.o`: Forward graph object file
- `subgraph.o`: Subgraph object file
- `transformer-runner`: Final executable

## Compilation Modes

### Staged Compilation Mode (Debugging/Analysis)

**Stage 1: Frontend (PyTorch → TOSA)**
- Target: `buddy-transformer-frontend`
- Converts PyTorch model to TOSA dialect MLIR representation
- Outputs: `forward.mlir`, `subgraph0.mlir`, `arg0.data`

**Stage 2: Midend (TOSA → Optimized MLIR)**
- Target: `buddy-transformer-midend`
- Applies optimization passes:
  - Buffer allocation and deallocation
  - Matrix multiplication optimization
  - Affine loop transformations
  - Parallelization
- Outputs: `forward-midend.mlir`, `subgraph0-midend.mlir`

**Stage 3: Backend (Optimized MLIR → LLVM MLIR)**
- Target: `buddy-transformer-backend`
- Lowers high-level operations to LLVM dialect:
  - Vector operations lowering
  - Memory operations finalization
  - Control flow conversion
- Outputs: `forward-backend.mlir`, `subgraph0-backend.mlir`

**Stage 4: Code Generation (LLVM MLIR → Object Files)**
- Target: `buddy-transformer-codegen`
- Generates final executable code:
  - LLVM IR generation: `forward.ll`, `subgraph0.ll`
  - Object file compilation: `forward.o`, `subgraph.o`
- Target: `buddy-transformer-executable`
- Links executable: `transformer-runner`

### One-step Compilation Mode (Production)

**Direct Compilation**
- Target: `buddy-transformer-onestep`
- Compiles directly from TOSA to object files without intermediate stages
- Faster build time, no intermediate files for inspection
- Outputs: `forward-onestep.o`, `subgraph-onestep.o`, `transformer-runner-onestep`

## Target Summary

| Target | Description | Output Files |
|--------|-------------|--------------|
| `buddy-transformer` | **One-step compilation** | `transformer-runner` |
| `buddy-transformer-frontend` | PyTorch → TOSA MLIR | `*.mlir`, `arg0.data` |
| `buddy-transformer-midend` | TOSA → Optimized MLIR | `*-midend.mlir` |
| `buddy-transformer-backend` | Optimized → LLVM MLIR | `*-backend.mlir` |
| `buddy-transformer-codegen` | LLVM MLIR → Object files | `*.ll`, `*.o` |
| `buddy-transformer-executable` | Staged executable | `transformer-runner-staged` |

