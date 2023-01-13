#!/bin/bash
BUDDY_LLC := ../../build/bin/buddy-llc

rv-buddy-mvin-asm:
	@${BUDDY_LLC} ./rv-buddy-mvin.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-mvin-mvout-asm:
	@${BUDDY_LLC} ./rv-buddy-mvin-mvout.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s

rv-buddy-mvout-mvout-obj:
	@${BUDDY_LLC} ./rv-buddy-mvin-mvout.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext,+D -filetype=obj --float-abi=hard\
		-o ./log.o
	
rv-buddy-mvin-mvout-exe:
	riscv64-unknown-linux-gnu-gcc  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
	-ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha -lm -lgcc  \
	-I$RISCV/../../generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests      \
	-I$RISCV/../../generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/env  \
	-I$RISCV/../../generators/gemmini/software/gemmini-rocc-tests/build/.. \
	-I$RISCV/../../generators/gemmini/software/gemmini-rocc-tests/build/../riscv-tests/benchmarks/common \
	-DID_STRING= -DPRINT_TILE=0  -static -DBAREMETAL=1 log.o -o mvin-mvout-pk

rv-buddy-mvin-mvout-run:
	spike --extension=gemmini pk mvin-mvout-pk
