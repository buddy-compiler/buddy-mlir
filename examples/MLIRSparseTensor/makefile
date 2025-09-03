MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-runner
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

SPARSE_MATRIX_A := ./data/sa.mtx
SPARSE_MATRIX_B := ./data/sb.mtx
SPARSE_MATRIX_C := ./data/generated.mtx

sparse-tensor-fuse-tensor-cast-lower:
	@${MLIR_OPT} ./sparse-tensor-fuse-tensor-cast.mlir\
		--pre-sparsification-rewrite -o ./log.mlir

sparse-tensor-new-lower:
	@${MLIR_OPT} ./sparse-tensor-new.mlir \
		--sparse-compiler="enable-runtime-library=true" -o log.mlir
sparse-tensor-new-translate:
	@${MLIR_OPT} ./sparse-tensor-new.mlir \
		--sparse-compiler="enable-runtime-library=true" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll
sparse-tensor-new-run:
	@${MLIR_OPT} ./sparse-tensor-new.mlir \
		--sparse-compiler="enable-runtime-library=true" | \
	TENSOR0=${SPARSE_MATRIX_A} TENSOR1=${SPARSE_MATRIX_B} TENSOR2=${SPARSE_MATRIX_C} \
	${MLIR_CPU_RUNNER} -e main -O0 --entry-point-result=void \
		--shared-libs=${MLIR_RUNNER_UTILS},${MLIR_C_RUNNER_UTILS}

sparse-tensor-pack-lower:
	@${MLIR_OPT} ./sparse-tensor-pack.mlir \
		--sparse-compiler="enable-runtime-library=false" -o log.mlir
sparse-tensor-pack-translate:
	@${MLIR_OPT} ./sparse-tensor-pack.mlir \
		--sparse-compiler="enable-runtime-library=false" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll
sparse-tensor-pack-run:
	@${MLIR_OPT} ./sparse-tensor-pack.mlir \
		--sparse-compiler="enable-runtime-library=false" | \
	${MLIR_CPU_RUNNER} -e main -O0 --entry-point-result=void \
		--shared-libs=${MLIR_RUNNER_UTILS},${MLIR_C_RUNNER_UTILS}

sparse-tensor-dump-lower:
	@${MLIR_OPT} ./sparse-tensor-dump.mlir \
		--sparse-compiler="enable-runtime-library=true" -o log.mlir
sparse-tensor-dump-translate:
	@${MLIR_OPT} ./sparse-tensor-dump.mlir \
		--sparse-compiler="enable-runtime-library=true" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll
sparse-tensor-dump-run:
	@${MLIR_OPT} ./sparse-tensor-dump.mlir \
		--sparse-compiler="enable-runtime-library=true" | \
	${MLIR_CPU_RUNNER} -e main -O0 --entry-point-result=void \
		--shared-libs=${MLIR_RUNNER_UTILS},${MLIR_C_RUNNER_UTILS}

sparse-tensor-number-of-entries-lower:
	@${MLIR_OPT} ./sparse-tensor-number-of-entries.mlir \
		--sparse-compiler="enable-runtime-library=false" -o log.mlir
sparse-tensor-number-of-entries-translate:
	@${MLIR_OPT} ./sparse-tensor-number-of-entries.mlir \
		--sparse-compiler="enable-runtime-library=false" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll
sparse-tensor-number-of-entries-run:
	@${MLIR_OPT} ./sparse-tensor-number-of-entries.mlir \
		--sparse-compiler="enable-runtime-library=false" | \
	${MLIR_CPU_RUNNER} -e main -O0 --entry-point-result=void \
		--shared-libs=${MLIR_RUNNER_UTILS},${MLIR_C_RUNNER_UTILS}

sparse-tensor-insert-lower:
	@${MLIR_OPT} ./sparse-tensor-insert.mlir \
		--sparse-compiler="enable-runtime-library=true" -o log.mlir
sparse-tensor-insert-translate:
	@${MLIR_OPT} ./sparse-tensor-insert.mlir \
		--sparse-compiler="enable-runtime-library=true" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll
sparse-tensor-insert-run:
	@${MLIR_OPT} ./sparse-tensor-insert.mlir \
		--sparse-compiler="enable-runtime-library=true" | \
	${MLIR_CPU_RUNNER} -e main -O0 --entry-point-result=void \
		--shared-libs=${MLIR_RUNNER_UTILS},${MLIR_C_RUNNER_UTILS}

sparse-tensor-binary-lower:
	@${MLIR_OPT} ./sparse-tensor-binary.mlir \
		--sparse-compiler="enable-runtime-library=true" -o log.mlir
sparse-tensor-binary-translate:
	@${MLIR_OPT} ./sparse-tensor-binary.mlir \
		--sparse-compiler="enable-runtime-library=true" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll
sparse-tensor-binary-run:
	@${MLIR_OPT} ./sparse-tensor-binary.mlir \
		--sparse-compiler="enable-runtime-library=true" | \
	${MLIR_CPU_RUNNER} -e main -O0 --entry-point-result=void \
		--shared-libs=${MLIR_RUNNER_UTILS},${MLIR_C_RUNNER_UTILS}

# `sparse_tensor.expand/compress` are used by rewrite op to lower those high level linalg operation.
# The below passes can make the rewrited code more readble.
sparse-tensor-expand-lower:
	@${MLIR_OPT} ./sparse-tensor-expand.mlir \
		--linalg-generalize-named-ops \
		--linalg-fuse-elementwise-ops \
		--sparsification -o log.mlir

# This target will show the original for-loop without vectorization,
# which is useful to compare with the vectorized version.
sparse-tensor-vectorization-linalg-lower:
	@${MLIR_OPT} ./sparse-tensor-vectorization.mlir \
		--linalg-generalize-named-ops \
		--linalg-fuse-elementwise-ops \
		--sparsification \
		-o log.mlir
sparse-tensor-vectorization-lower:
	@${MLIR_OPT} ./sparse-tensor-vectorization.mlir \
		--sparsification --cse \
		--sparse-vectorization="vl=16" --cse \
		-o log.mlir
# This example is used for code verification only, as there is currently no ARMSVE machine for us to run the code on.
# Do the same run, but with VLA enable
sparse-tensor-vla-vectorization-lower:
	@${MLIR_OPT} ./sparse-tensor-vectorization.mlir \
		--sparsification --cse \
		--sparse-vectorization="vl=16 enable-vla-vectorization=true" --cse \
		-o log.mlir
