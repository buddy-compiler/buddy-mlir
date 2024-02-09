mlir-opt matmul.mlir -test-transform-dialect-interpreter="transform-file-name=transform.mlir" -o stage1.mlir
mlir-opt stage1.mlir -test-transform-dialect-interpreter="transform-file-name=gpu-transform.mlir" -o stage2.mlir
mlir-opt stage2.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -linalg-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize -canonicalize -o stage3.mlir
mlir-opt stage3.mlir -canonicalize -gpu-kernel-outlining -canonicalize -o stage4.mlir
buddy-opt stage4.mlir -convert-memcpy-to-gpu -o matmul-converted.mlir