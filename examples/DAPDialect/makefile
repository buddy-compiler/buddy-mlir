#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0
CLANG := ../../llvm/build//bin/clang
MLIR_LIB := ../../llvm/build/lib/
BUDDY_LIB := ../../build/lib/

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MLIR_ASYNC_RUNTIME := ./../llvm/build/lib/libmlir_async_runtime.dylib
MTRIPLE := x86_64-apple-darwin
endif

dap-parallel-lower:
	@${BUDDY_OPT} ./test.mlir \
	    -lower-dap="DAP-vector-splitting=128" \
		-async-parallel-for \
		-async-to-async-runtime \
		-async-runtime-ref-counting \
        -async-runtime-ref-counting-opt \
		-arith-expand \
		-convert-async-to-llvm \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-scf-to-cf -convert-vector-to-llvm \
		--llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-convert-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts \
		-o ./log.mlir

dap-parallel-translate:
	@${BUDDY_OPT} ./test.mlir \
	    -lower-dap="DAP-vector-splitting=128" \
		-async-parallel-for \
		-async-to-async-runtime \
		-async-runtime-ref-counting \
        -async-runtime-ref-counting-opt \
		-arith-expand \
		-convert-async-to-llvm \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-scf-to-cf -convert-vector-to-llvm \
		--llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-convert-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

dap-parallel-aot:
	@${BUDDY_OPT} ./test.mlir \
	    -lower-dap="DAP-vector-splitting=128" \
		-async-parallel-for \
		-async-to-async-runtime \
		-async-runtime-ref-counting \
        -async-runtime-ref-counting-opt \
		-arith-expand \
		-convert-async-to-llvm \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-scf-to-cf -convert-vector-to-llvm \
		--llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-convert-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll
	@${CLANG} ${OPT_FLAG} log.ll \
		-L ${MLIR_LIB} \
		-L ${BUDDY_LIB} \
		-lmlir_runner_utils -lmlir_c_runner_utils \
		-lstatic_mlir_async_runtime \
		-lLLVMSupport \
		-lstdc++ -lpthread -ltinfo -lm \
		-o a.out
	@LD_LIBRARY_PATH=${MLIR_LIB} ./a.out

dap-parallel-single-aot:
	@${BUDDY_OPT} ./test.mlir \
	 	-lower-dap="DAP-vector-splitting=128" \
		-arith-expand \
        -convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-scf-to-cf -convert-vector-to-llvm \
		--llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-convert-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll
	@${CLANG} ${OPT_FLAG} log.ll \
		-L ${MLIR_LIB} \
		-L ${BUDDY_LIB} \
		-lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${MLIR_LIB} ./a.out

dap-buddy-parallel-single-aot:
	@${BUDDY_OPT} ./biquad.mlir \
	 	-lower-dap="DAP-vector-splitting=128" \
		-arith-expand \
        -convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-scf-to-cf -convert-vector-to-llvm \
		--llvm-request-c-wrappers \
		-convert-arith-to-llvm \
		-convert-memref-to-llvm \
		-convert-func-to-llvm \
		-reconcile-unrealized-casts |\
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll
	@${CLANG} ${OPT_FLAG} log.ll \
		-L ${MLIR_LIB} \
		-L ${BUDDY_LIB} \
		-lmlir_runner_utils -lmlir_c_runner_utils \
		-o a.out
	@LD_LIBRARY_PATH=${MLIR_LIB} ./a.out