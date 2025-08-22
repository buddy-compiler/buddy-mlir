# Buddy Compiler DeepSeek R1 Attention Example

## Introduction
This example demonstrates how to use Buddy Compiler to compile a single layer of DeepSeek R1 1.5B attention mechanism to optimized MLIR code. The compilation pipeline includes frontend conversion from PyTorch to MLIR and midend optimization passes.

## Files Structure
- `attention_model.py`: DeepSeek R1 single layer attention PyTorch model definition
- `import-attention.py`: Frontend script to convert PyTorch model to MLIR
- `CMakeLists.txt`: Build configuration for frontend and midend compilation
- `README.md`: This documentation

## Model Configuration
- Hidden size: 1536
- Number of attention heads: 12
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

1. Enable the attention example in CMake:
```bash
cd buddy-mlir/build
cmake -G Ninja .. -DBUDDY_ATTENTION_EXAMPLES=ON
```

2. Build the attention example:

For complete build (all stages):
```bash
ninja buddy-attention
```

For frontend only (PyTorch to TOSA):
```bash
ninja buddy-attention-frontend
```

For midend optimization only:
```bash
ninja buddy-attention-midend
```

For backend optimization only (requires midend):
```bash
ninja buddy-attention-backend
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
- `forward-midend.mlir`: Midend optimized main graph (before LLVM lowering)
- `subgraph0-midend.mlir`: Midend optimized subgraph (before LLVM lowering)

### Backend Output (LLVM Optimized MLIR)
- `forward-optimized.mlir`: Backend optimized main graph (LLVM dialect)
- `subgraph0-optimized.mlir`: Backend optimized subgraph (LLVM dialect)

## Compilation Pipeline

### Frontend (Stage 0) - `ninja buddy-attention-frontend`
1. PyTorch model definition (`attention_model.py`)
2. Model export using DynamoCompiler (`import-attention.py`)
3. Buddy Graph representation (`graph.log`, `graph_fused.log`)
4. TOSA dialect MLIR generation (`forward.mlir`, `subgraph0.mlir`)

### Midend Optimization (Stage 1) - `ninja buddy-attention-midend`
1. TOSA to Linalg conversion
2. Arithmetic expansion
3. Tensor optimization
4. Bufferization
5. Matrix multiplication optimization
6. Affine loop transformations
7. Parallelization
8. Vector optimization

### Backend Optimization (Stage 2) - `ninja buddy-attention-backend`
1. Vector to LLVM conversion
2. Memory reference expansion
3. Arithmetic to LLVM conversion
4. Control flow to LLVM conversion
5. Function to LLVM conversion
6. Final reconciliation

## Customization

### Input Parameters
Modify the import script parameters:
```bash
python import-attention.py --batch-size 2 --seq-len 128
```

### Model Configuration
Edit `attention_model.py` to change model parameters:
- `hidden_size`: Model hidden dimension
- `num_attention_heads`: Number of attention heads
- `head_dim`: Dimension per attention head

## Testing
Test the model independently:
```bash
cd examples/BuddyAttention
python attention_model.py
```

## Notes
- This example focuses on frontend and midend compilation only
- Backend LLVM code generation is not included
- The attention implementation follows DeepSeek R1 1.5B architecture
- Uses f32 precision for optimal compatibility
