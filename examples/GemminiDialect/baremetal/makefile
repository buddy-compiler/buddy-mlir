#!/bin/bash
BUDDY_MLIR_DIR := ../../../
BUDDY_OPT := ${BUDDY_MLIR_DIR}/build/bin/buddy-opt
BUDDY_TRANSLATE := ${BUDDY_MLIR_DIR}/build/bin/buddy-translate
BUDDY_LLC := ${BUDDY_MLIR_DIR}/build/bin/buddy-llc

GEMMINI_EXAMPLE_DIR := ${BUDDY_MLIR_DIR}/examples/GemminiDialect/

CC := riscv64-unknown-elf-gcc

mvin-mvout-run-baremetal:
	@${BUDDY_OPT} ./${GEMMINI_EXAMPLE_DIR}/mvin-mvout.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-relocation-model=pic \
		-o log.o
	@${CC} -static -specs=htif_nano.specs log.o -o mvin-mvout-baremetal
	@spike --extension=gemmini mvin-mvout-baremetal

gemmini-conv2d-nchw-fchw-f32-run-baremetal:
	@${BUDDY_OPT} ./${GEMMINI_EXAMPLE_DIR}/conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-relocation-model=pic \
		-o log.o
	@${CC} -O2 -static -specs=htif_nano.specs log.o -o conv-2d-nchw-fchw-f32-baremetal
	@spike --extension=gemmini conv-2d-nchw-fchw-f32-baremetal

gemmini-conv2d-nchw-fchw-i8-run-baremetal:
	@${BUDDY_OPT} ./${GEMMINI_EXAMPLE_DIR}/conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-relocation-model=pic \
		-o log.o
	@${CC} -O2 -static -specs=htif_nano.specs log.o -o conv-2d-nchw-fchw-i8-baremetal
	@spike --extension=gemmini conv-2d-nchw-fchw-i8-baremetal

gemmini-conv2d-nchw-fhwc-f32-run-baremetal:
	@${BUDDY_OPT} ./${GEMMINI_EXAMPLE_DIR}/conv_2d_nchw_fhwc_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-relocation-model=pic \
		-o log.o
	@${CC} -O2 -static -specs=htif_nano.specs log.o -o conv-2d-nchw-fhwc-f32-baremetal
	@spike --extension=gemmini conv-2d-nchw-fhwc-f32-baremetal

gemmini-conv2d-nhwc-fhwc-i8-run-baremetal:
	@${BUDDY_OPT} ./${GEMMINI_EXAMPLE_DIR}/conv_2d_nhwc_fhwc_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-relocation-model=pic \
		-o log.o
	@${CC} -O2 -static -specs=htif_nano.specs log.o -o conv-2d-nhwc-fhwc-i8-baremetal
	@spike --extension=gemmini conv-2d-nhwc-fhwc-i8-baremetal
