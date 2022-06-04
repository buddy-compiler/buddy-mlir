#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
MLIR_TOYC := ../../llvm/build/bin/toyc-ch7

toyc-constant-run:
	@${MLIR_TOYC} ./constant.toy -emit=jit
