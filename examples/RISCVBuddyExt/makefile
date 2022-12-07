#!/bin/bash
BUDDY_LLC := ../../build/bin/buddy-llc

rv-buddy-mvin-asm:
	@${BUDDY_LLC} ./rv-buddy-mvin.ll \
		-mtriple=riscv64 -verify-machineinstrs -mattr=+buddyext \
		-o ./log.s
