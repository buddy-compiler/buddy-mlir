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

scf-while-lower:
	@${MLIR_OPT} ./scf-while.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

scf-while-translate:
	@${MLIR_OPT} ./scf-while.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

scf-while-run:
	@${MLIR_OPT} ./scf-while.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

scf-parallel-lower:
	@${MLIR_OPT} ./scf-parallel.mlir \
	    -async-parallel-for \
		-o ./log.mlir

scf-parallel-translate:
	@${MLIR_OPT} ./scf-parallel.mlir \
	    -async-parallel-for \
        -async-to-async-runtime \
		-async-runtime-ref-counting \
        -async-runtime-ref-counting-opt \
		-arith-expand \
        -convert-async-to-llvm \
        -convert-scf-to-cf \
        -convert-memref-to-llvm \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts |\
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

scf-parallel-run:
	@${MLIR_OPT} ./scf-parallel.mlir \
	    -async-parallel-for \
        -async-to-async-runtime \
		-async-runtime-ref-counting \
        -async-runtime-ref-counting-opt \
		-arith-expand \
        -convert-async-to-llvm \
        -convert-scf-to-cf \
        -convert-memref-to-llvm \
        -convert-func-to-llvm \
        -reconcile-unrealized-casts |\
    ${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
	    -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_ASYNC_RUNTIME}

scf-parallel-aot:
	@${MLIR_OPT} ./scf-parallel.mlir \
	    -async-parallel-for \
        -async-to-async-runtime \
		-async-runtime-ref-counting \
        -async-runtime-ref-counting-opt \
		-arith-expand \
        -convert-async-to-llvm \
        -convert-scf-to-cf \
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

scf-for-lower:
	@${MLIR_OPT} ./scf-for.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

scf-for-translate:
	@${MLIR_OPT} ./scf-for.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

scf-for-run:
	@${MLIR_OPT} ./scf-for.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}

scf-if-lower:
	@${MLIR_OPT} ./scf-if.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

scf-if-translate:
	@${MLIR_OPT} ./scf-if.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

scf-if-run:
	@${MLIR_OPT} ./scf-if.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}	

scf-do-while-lower:
	@${MLIR_OPT} ./scf-do-while.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts -o ./log.mlir

scf-do-while-translate:
	@${MLIR_OPT} ./scf-do-while.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

scf-do-while-run:
	@${MLIR_OPT} ./scf-do-while.mlir \
		--convert-scf-to-cf --convert-vector-to-llvm --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
