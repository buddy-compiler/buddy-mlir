mlir-opt test.mlir -arith-expand -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -linalg-bufferize -convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine -canonicalize -func-bufferize -arith-bufferize -tensor-bufferize -buffer-deallocation -finalizing-bufferize -canonicalize -o bufferized.mlir
mlir-opt bufferized.mlir -gpu-map-parallel-loops -convert-parallel-loops-to-gpu -canonicalize -gpu-kernel-outlining -o outlined.mlir
buddy-opt outlined.mlir -gpu-host-register -o host-registered.mlir
mlir-opt host-registered.mlir -convert-scf-to-cf -memref-expand -finalize-memref-to-llvm -convert-arith-to-llvm -convert-gpu-to-nvvm='has-redux=1' -o nvvm.mlir
mlir-opt nvvm.mlir -llvm-request-c-wrappers -o wrapper.mlir
mlir-opt wrapper.mlir --test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" -o cubin.mlir
mlir-cpu-runner cubin.mlir -entry-point-result=void -shared-libs=/home/liam/IPRC/llvm-project/build/lib/libmlir_runner_utils.so -shared-libs=/home/liam/IPRC/llvm-project/build/lib/libmlir_cuda_runtime.so