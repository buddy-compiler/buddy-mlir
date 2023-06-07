MLIR_OPT := ../../llvm/build/bin/mlir-opt

sparse-tensor-fuse-tensor-cast-lower:
	@${MLIR_OPT} ./sparse-tensor-fuse-tensor-cast.mlir\
		--pre-sparsification-rewrite -o ./log.mlir
