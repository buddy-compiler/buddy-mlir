#!/bin/bash
BUDDY_LLC := ../../build/bin/buddy-llc

rv-buddy-mvin-mvout-asm:
	@${BUDDY_LLC} ./rv-buddy-mvin-mvout.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-mvin-mvout-obj:
	@${BUDDY_LLC} ./rv-buddy-mvin-mvout.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-buddy-mvin-mvout-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o mvin-mvout-pk

rv-buddy-mvin-mvout-run:
	spike --extension=gemmini pk mvin-mvout-pk

rv-buddy-matrix-add-asm:
	@${BUDDY_LLC} ./rv-buddy-matrix-add.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-matrix-add-obj:
	@${BUDDY_LLC} ./rv-buddy-matrix-add.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-buddy-matrix-add-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matrix-add-pk

rv-buddy-matrix-add-run:
	spike --extension=gemmini pk matrix-add-pk

rv-buddy-transpose-asm:
	@${BUDDY_LLC} ./rv-buddy-transpose.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-transpose-obj:
	@${BUDDY_LLC} ./rv-buddy-transpose.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-buddy-transpose-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o transpose-pk

rv-buddy-transpose-run:
	spike --extension=gemmini pk transpose-pk

rv-buddy-mvin-padded-asm:
	@${BUDDY_LLC} ./rv-buddy-mvin-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-mvin-padded-obj:
	@${BUDDY_LLC} ./rv-buddy-mvin-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-buddy-mvin-padded-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o mvin-padded-pk

rv-buddy-mvin-padded-run:
	spike --extension=gemmini pk mvin-padded-pk

rv-buddy-mvout-padded-asm:
	@${BUDDY_LLC} ./rv-buddy-mvout-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-mvout-padded-obj:
	@${BUDDY_LLC} ./rv-buddy-mvout-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-buddy-mvout-padded-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o mvout-padded-pk

rv-buddy-mvout-padded-run:
	spike --extension=gemmini pk mvout-padded-pk

rv-buddy-matmuls-padded-asm:
	@${BUDDY_LLC} ./rv-buddy-matmuls-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-matmuls-padded-obj:
	@${BUDDY_LLC} ./rv-buddy-matmuls-padded.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-buddy-matmuls-padded-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o matmuls-padded-pk

rv-buddy-matmuls-padded-run:
	spike --extension=gemmini pk matmuls-padded-pk

rv-buddy-tile-matmul-ws-asm:
	@${BUDDY_LLC} ./rv-buddy-tile-matmul-ws.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-tile-matmul-ws-obj:
	@${BUDDY_LLC} ./rv-buddy-tile-matmul-ws.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-buddy-tile-matmul-ws-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o tile-matmul-ws-pk -lm

rv-buddy-tile-matmul-ws-run:
	spike --extension=gemmini pk tile-matmul-ws-pk

rv-buddy-conv-asm:
	@${BUDDY_LLC} ./rv-buddy-conv.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-conv-obj:
	@${BUDDY_LLC} ./rv-buddy-conv.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-buddy-conv-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lgcc  \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o conv-pk -lm

rv-buddy-conv-run:
	spike --extension=gemmini pk conv-pk
