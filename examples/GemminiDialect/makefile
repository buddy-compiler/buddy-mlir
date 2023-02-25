#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc 

mvin-mvout-lower:
	@${BUDDY_OPT} ./mvin-mvout.mlir \
		--lower-gemmini -o ./log.mlir

mvin-mvout-translate:
	@${BUDDY_OPT} ./mvin-mvout.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll 

mvin-mvout-obj:
	@${BUDDY_OPT} ./mvin-mvout.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir | \
		${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 

mvin-mvout-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o mvin-mvout-pk

mvin-mvout-run:
	spike --extension=gemmini pk mvin-mvout-pk
	