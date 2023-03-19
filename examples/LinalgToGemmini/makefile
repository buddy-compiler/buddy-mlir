#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc 

matmul-lower:
	@${BUDDY_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini -o ./log.mlir 
	
matmul-translate:
	@${BUDDY_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | ${BUDDY_TRANSLATE} \
		--buddy-to-llvmir -o log.ll 

matmul-obj:
	@${BUDDY_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | ${BUDDY_TRANSLATE} \
		--buddy-to-llvmir | ${BUDDY_LLC} -filetype=obj \
		-mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 
	
matmul-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matmul-pk

matmul-run:
	spike --extension=gemmini pk matmul-pk

conv_2d_nchw_fchw-lower:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini -o ./log.mlir 
	
conv_2d_nchw_fchw-translate:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | ${BUDDY_TRANSLATE} \
		--buddy-to-llvmir -o log.ll 

conv_2d_nchw_fchw-obj:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | ${BUDDY_TRANSLATE} \
		--buddy-to-llvmir | ${BUDDY_LLC} -filetype=obj \
		-mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 
	
conv_2d_nchw_fchw-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o conv_2d_nchw_fchw-pk

conv_2d_nchw_fchw-run:
	spike --extension=gemmini pk conv_2d_nchw_fchw-pk
