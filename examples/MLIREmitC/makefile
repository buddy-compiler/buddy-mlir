#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate

emitc-include-emit:
	@${MLIR_TRANSLATE} ./emitc-include.mlir \
		-mlir-to-cpp \
		-o log.cpp
