#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt

bud-array-attr-lower:
	@${BUDDY_OPT} ./bud-array-attr.mlir \
		--lower-bud -o ./log.mlir

bud-constant-lower:
	@${BUDDY_OPT} ./bud-constant.mlir \
		--lower-bud -o ./log.mlir

bud-print-lower:
	@${BUDDY_OPT} ./bud-print.mlir \
		--lower-bud -o ./log.mlir

bud-str-attr-lower:
	@${BUDDY_OPT} ./bud-str-attr.mlir \
		--lower-bud -o ./log.mlir

bud-inline-interface-lower:
	@${BUDDY_OPT} ./bud-inline-interface.mlir \
		--inline -o ./log.mlir

# TODO: all fails
test-all:
	@echo "TODO"
# 	bud-array-attr-lower \
# 	bud-constant-lower \
# 	bud-print-lower \
# 	bud-str-attr-lower \
# 	bud-vector-config-lower \
# 	bud-inline-interface-lower
