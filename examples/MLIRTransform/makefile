#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
CLANG := ../../llvm/build/bin/clang
OPT_FLAG := -O0

MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MLIR_CUDA_RUNTIME := ../../llvm/build/lib/libmlir_cuda_runtime.so

transform-conv2d-im2col-lower:
	@${MLIR_OPT} ./transform-conv2d-im2col.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--linalg-bufferize \
		--convert-linalg-to-loops \
		--func-bufferize \
		--arith-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		-o log.mlir

transform-batch-matmul-e2e-lower:
	@${MLIR_OPT} ./transform-batch-matmul-e2e.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		-o log.mlir

transform-batch-matmul-e2e-x86-asm:
	@${MLIR_OPT} ./transform-batch-matmul-e2e.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--finalize-memref-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f\
		--filetype=asm -o log.s

transform-batch-matmul-e2e-rv-asm:
	@${MLIR_OPT} ./transform-batch-matmul-e2e.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--finalize-memref-to-llvm \
		--convert-math-to-llvm \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -riscv-v-vector-bits-min=128 \
		--filetype=asm -o log.s

transform-batch-matmul-e2e-x86-o:
	@${MLIR_OPT} ./transform-batch-matmul-e2e.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--finalize-memref-to-llvm \
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
 	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple=x86_64-unknown-linux-gnu \
		-mattr=avx512f --filetype=obj \
	    -o ./transform-batch-matmul-e2e.o

transform-batch-matmul-e2e-rv-o:
	@${MLIR_OPT} ./transform-batch-matmul-e2e.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--convert-linalg-to-llvm \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--finalize-memref-to-llvm \
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} ${OPT_FLAG} -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v -filetype=obj -riscv-v-vector-bits-min=128 \
		-o ./transform-batch-matmul-e2e.o

transform-gpu-nested-forall-lower:
	@${MLIR_OPT} ./gpu-nested-forall.mlir \
		--test-transform-dialect-interpreter \
		--test-transform-dialect-erase-schedule \
		--canonicalize | \
	@${MLIR_OPT} \
	  --gpu-kernel-outlining \
		--convert-gpu-to-nvvm \
		--reconcile-unrealized-casts | \
	@${MLIR_OPT} \
	 	--llvm-request-c-wrappers | \
	@${MLIR_OPT} \
		--test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" -o gpu-nested-forall.mlir

transform-gpu-reduction-tile-first-step:
	@${MLIR_OPT} ./gpu-nested-forall.mlir \
			--test-transform-dialect-interpreter \
			--test-transform-dialect-erase-schedule \
			--canonicalize | \
	@${MLIR_OPT} \
			-one-shot-bufferize="bufferize-function-boundaries" \
			-o gpu-reduction-tile-first-step.mlir

transform-gpu-reduction-tile-second-step:
	@${MLIR_OPT} ./gpu-reduction-tile-first-step.mlir \
			--test-transform-dialect-interpreter \
			--test-transform-dialect-erase-schedule \
			--canonicalize | \
	@${MLIR_OPT} \
	  --gpu-kernel-outlining \
		--convert-gpu-to-nvvm \
		--reconcile-unrealized-casts | \
	@${MLIR_OPT} \
	 	--llvm-request-c-wrappers | \
	@${MLIR_OPT} \
		--test-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx71 cubin-format=fatbin" \
		-o gpu-nested-forall.mlir
