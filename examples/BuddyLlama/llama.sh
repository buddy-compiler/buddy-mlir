buddy-opt llama-linalg-default.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -linalg-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize -canonicalize -o llama-bufferized.mlir
# mlir-opt llama-linalg-default.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries copy-before-write" -expand-realloc  -resolve-shaped-type-result-dims -canonicalize -buffer-deallocation-simplification -bufferization-lower-deallocations -cse -canonicalize -buffer-deallocation-pipeline -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize  -o llama-bufferized.mlir
buddy-opt llama-bufferized.mlir -gpu-map-parallel-loops -convert-parallel-loops-to-gpu -canonicalize -gpu-kernel-outlining -o llama-outlined.mlir
buddy-opt llama-outlined.mlir -gpu-host-register -o llama-host-registered.mlir
buddy-opt llama-host-registered.mlir -convert-scf-to-cf -memref-expand -finalize-memref-to-llvm -convert-arith-to-llvm -convert-gpu-to-nvvm='has-redux=1' -o llama-nvvm.mlir
mlir-opt llama-nvvm.mlir -llvm-request-c-wrappers -o llama-wrapper.mlir
mlir-opt llama-wrapper.mlir --test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" -o llama-cubin.mlir
mlir-translate llama-cubin.mlir --mlir-to-llvmir -o llama.ll
/home/liam/IPRC/llvm-project/build/bin/llc llama.ll -filetype=obj -relocation-model=pic -O3 -o llama.o
clang llama.o llama-main.cpp.o /home/liam/IPRC/llvm-project/build/lib/libmlir_cuda_runtime.so /home/liam/IPRC/llvm-project/build/lib/libmlir_c_runner_utils.so -lstdc++ -o llama.out
./llama.out