mlir-opt matmul.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -linalg-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize -canonicalize -o matmul-bufferized.mlir
mlir-opt matmul-bufferized.mlir -gpu-map-parallel-loops -convert-parallel-loops-to-gpu -canonicalize -gpu-kernel-outlining -canonicalize -o matmul-outlined.mlir
buddy-opt matmul-outlined.mlir  -convert-memcpy-to-gpu -o matmul-converted.mlir
buddy-opt matmul-outlined.mlir  -convert-memcpy-to-gpu -gpu-async-region -o matmul-converted.mlir
buddy-opt matmul-converted.mlir -convert-scf-to-cf -memref-expand -finalize-memref-to-llvm -convert-arith-to-llvm -convert-gpu-to-nvvm='has-redux=1' -o matmul-nvvm.mlir
mlir-opt matmul-nvvm.mlir -llvm-request-c-wrappers -o matmul-wrapper.mlir
mlir-opt matmul-wrapper.mlir --test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" -o matmul-cubin.mlir
python3 run-module-gpu.py