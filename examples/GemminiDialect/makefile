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

matmul-os-lower:
	@${BUDDY_OPT} ./matmul-os.mlir \
		--lower-gemmini -o ./log.mlir

matmul-os-translate:
	@${BUDDY_OPT} ./matmul-os.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll 

matmul-os-obj:
	@${BUDDY_OPT} ./matmul-os.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir | \
		${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 

matmul-os-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matmul-os-pk 

matmul-os-run:
	spike --extension=gemmini pk matmul-os-pk  

compute-accumulated-lower:
	@${BUDDY_OPT} ./compute-accumulated.mlir \
		--lower-gemmini -o ./log.mlir

compute-accumulated-translate:
	@${BUDDY_OPT} ./compute-accumulated.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll 

compute-accumulated-obj:
	@${BUDDY_OPT} ./compute-accumulated.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir | \
		${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 

compute-accumulated-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o compute-accumulated-pk 

compute-accumulated-run:
	spike --extension=gemmini pk compute-accumulated-pk  

matmul-ws-lower:
	@${BUDDY_OPT} ./matmul-ws.mlir \
		--lower-gemmini -o ./log.mlir

matmul-ws-translate:
	@${BUDDY_OPT} ./matmul-ws.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll 

matmul-ws-obj:
	@${BUDDY_OPT} ./matmul-ws.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir | \
		${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 

matmul-ws-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matmul-ws-pk 

matmul-ws-run:
	spike --extension=gemmini pk matmul-ws-pk  

tile-matmul-lower:
	@${BUDDY_OPT} ./tile-matmul.mlir \
		--lower-gemmini -o ./log.mlir

tile-matmul-translate:
	@${BUDDY_OPT} ./tile-matmul.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll 

tile-matmul-obj:
	@${BUDDY_OPT} ./tile-matmul.mlir \
		--lower-gemmini | ${BUDDY_TRANSLATE} --buddy-to-llvmir | \
		${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D \
		--float-abi=hard -o log.o 

tile-matmul-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o tile-matmul-pk 

tile-matmul-run:
	spike --extension=gemmini pk tile-matmul-pk 
