#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt

buddy-gpu-matmul-lower:
	@${BUDDY_OPT} matmul.mlir \
		-transform-preload-library="transform-library-paths=transform.mlir" \
		-transform-interpreter="entry-point=codegen" \
		-o log.mlir
