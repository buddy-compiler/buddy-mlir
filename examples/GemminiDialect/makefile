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

matrix-add-lower:
	@${BUDDY_OPT} ./matrix-add.mlir \
		--lower-gemmini -o ./log.mlir

matrix-add-translate:
	@${BUDDY_OPT} ./matrix-add.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll 

matrix-add-obj:
	@${BUDDY_OPT} ./matrix-add.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir | \
		${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 

matrix-add-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matrix-add-pk 

matrix-add-run:
	spike --extension=gemmini pk matrix-add-pk 

matrix-add-scale-lower:
	@${BUDDY_OPT} ./matrix-add-scale.mlir \
		--lower-gemmini -o ./log.mlir

matrix-add-scale-translate:
	@${BUDDY_OPT} ./matrix-add-scale.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll 

matrix-add-scale-obj:
	@${BUDDY_OPT} ./matrix-add-scale.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir | \
		${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 

matrix-add-scale-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matrix-add-scale-pk 

matrix-add-scale-run:
	spike --extension=gemmini pk matrix-add-scale-pk 

transpose-lower:
	@${BUDDY_OPT} ./transpose.mlir \
		--lower-gemmini -o ./log.mlir

transpose-translate:
	@${BUDDY_OPT} ./transpose.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll 

transpose-obj:
	@${BUDDY_OPT} ./transpose.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir | \
		${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 

transpose-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o transpose-pk 

transpose-run:
	spike --extension=gemmini pk transpose-pk

matmul-lower:
	@${BUDDY_OPT} ./matmul.mlir \
		--lower-gemmini -o ./log.mlir

matmul-translate:
	@${BUDDY_OPT} ./matmul.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll 

matmul-obj:
	@${BUDDY_OPT} ./matmul.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir | \
		${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 

matmul-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matmul-pk 

matmul-run:
	spike --extension=gemmini pk matmul-pk  
