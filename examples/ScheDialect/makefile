#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt

sche-addf-tensor-lower:
	@${BUDDY_OPT} ./sche-addf-tensor.mlir \
		--device-schedule \
		--lower-sche -o ./log.mlir

sche-addf-vector-lower:
	@${BUDDY_OPT} ./sche-addf-vector.mlir \
		--device-schedule \
		--lower-sche -o ./log.mlir

sche-scf-for-lower:
	@${BUDDY_OPT} ./sche-scf-for.mlir \
		--device-schedule -o ./log.mlir