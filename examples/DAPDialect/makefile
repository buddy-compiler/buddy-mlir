#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt

dap-fir-lower:
	@${BUDDY_OPT} ./fir.mlir \
		--lower-dap -o ./log.mlir
