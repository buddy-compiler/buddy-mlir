#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc 
INTERFACES:= ../../frontend/Interfaces

mvin-mvout-lower:
	@${BUDDY_OPT} ./mvin-mvout.mlir \
		-lower-gemmini \
		-o log.mlir

mvin-mvout-translate:
	@${BUDDY_OPT} ./mvin-mvout.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

mvin-mvout-asm:
	@${BUDDY_OPT} ./mvin-mvout.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

mvin-mvout-run:
	@${BUDDY_OPT} ./mvin-mvout.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matrix-add-lower:
	@${BUDDY_OPT} ./matrix-add.mlir \
		-lower-gemmini \
		-o log.mlir

matrix-add-translate:
	@${BUDDY_OPT} ./matrix-add.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

matrix-add-asm:
	@${BUDDY_OPT} ./matrix-add.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

matrix-add-run:
	@${BUDDY_OPT} ./matrix-add.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matrix-add-scale-lower:
	@${BUDDY_OPT} ./matrix-add-scale.mlir \
		-lower-gemmini \
		-o log.mlir

matrix-add-scale-translate:
	@${BUDDY_OPT} ./matrix-add-scale.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

matrix-add-scale-asm:
	@${BUDDY_OPT} ./matrix-add-scale.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

matrix-add-scale-run:
	@${BUDDY_OPT} ./matrix-add-scale.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

transpose-lower:
	@${BUDDY_OPT} ./transpose.mlir \
		-lower-gemmini \
		-o log.mlir

transpose-translate:
	@${BUDDY_OPT} ./transpose.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

transpose-asm:
	@${BUDDY_OPT} ./transpose.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

transpose-run:
	@${BUDDY_OPT} ./transpose.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matmul-os-lower:
	@${BUDDY_OPT} ./matmul-os.mlir \
		-lower-gemmini \
		-o log.mlir

matmul-os-translate:
	@${BUDDY_OPT} ./matmul-os.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

matmul-os-asm:
	@${BUDDY_OPT} ./matmul-os.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

matmul-os-run:
	@${BUDDY_OPT} ./matmul-os.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

compute-accumulated-lower:
	@${BUDDY_OPT} ./compute-accumulated.mlir \
		-lower-gemmini \
		-o log.mlir

compute-accumulated-translate:
	@${BUDDY_OPT} ./compute-accumulated.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

compute-accumulated-asm:
	@${BUDDY_OPT} ./compute-accumulated.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

compute-accumulated-run:
	@${BUDDY_OPT} ./compute-accumulated.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matmul-ws-lower:
	@${BUDDY_OPT} ./matmul-ws.mlir \
		-lower-gemmini \
		-o log.mlir

matmul-ws-translate:
	@${BUDDY_OPT} ./matmul-ws.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

matmul-ws-asm:
	@${BUDDY_OPT} ./matmul-ws.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

matmul-ws-run:
	@${BUDDY_OPT} ./matmul-ws.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-lower:
	@${BUDDY_OPT} ./tile-matmul.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-translate:
	@${BUDDY_OPT} ./tile-matmul.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-matmul-asm:
	@${BUDDY_OPT} ./tile-matmul.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s	

tile-matmul-run:
	@${BUDDY_OPT} ./tile-matmul.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-os-lower:
	@${BUDDY_OPT} ./tile-matmul-os.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-os-translate:
	@${BUDDY_OPT} ./tile-matmul-os.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-matmul-os-asm:
	@${BUDDY_OPT} ./tile-matmul-os.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-matmul-os-run:
	@${BUDDY_OPT} ./tile-matmul-os.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-igelu-lower:
	@${BUDDY_OPT} ./tile-matmul-ws-igelu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-igelu-translate:
	@${BUDDY_OPT} ./tile-matmul-ws-igelu.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-matmul-ws-igelu-asm:
	@${BUDDY_OPT} ./tile-matmul-ws-igelu.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-igelu-run:
	@${BUDDY_OPT} ./tile-matmul-ws-igelu.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-relu-lower:
	@${BUDDY_OPT} ./tile-matmul-ws-relu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-relu-translate:
	@${BUDDY_OPT} ./tile-matmul-ws-relu.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-matmul-ws-relu-asm:
	@${BUDDY_OPT} ./tile-matmul-ws-relu.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-relu-run:
	@${BUDDY_OPT} ./tile-matmul-ws-relu.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-softmax-lower:
	@${BUDDY_OPT} ./tile-matmul-ws-softmax.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-softmax-translate:
	@${BUDDY_OPT} ./tile-matmul-ws-softmax.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-matmul-ws-softmax-asm:
	@${BUDDY_OPT} ./tile-matmul-ws-softmax.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-softmax-run:
	@${BUDDY_OPT} ./tile-matmul-ws-softmax.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-layernorm-lower:
	@${BUDDY_OPT} ./tile-matmul-ws-layernorm.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-layernorm-translate:
	@${BUDDY_OPT} ./tile-matmul-ws-layernorm.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-matmul-ws-layernorm-asm:
	@${BUDDY_OPT} ./tile-matmul-ws-layernorm.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-layernorm-run:
	@${BUDDY_OPT} ./tile-matmul-ws-layernorm.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-lower:
	@${BUDDY_OPT} ./tile-conv.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-translate:
	@${BUDDY_OPT} ./tile-conv.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-conv-asm:
	@${BUDDY_OPT} ./tile-conv.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-conv-run:
	@${BUDDY_OPT} ./tile-conv.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

dw-tile-conv-lower:
	@${BUDDY_OPT} ./dw-tile-conv.mlir \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

dw-tile-conv-translate:
	@${BUDDY_OPT} ./dw-tile-conv.mlir \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

dw-tile-conv-asm:
	@${BUDDY_OPT} ./dw-tile-conv.mlir \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

dw-tile-conv-run:
	@${BUDDY_OPT} ./dw-tile-conv.mlir -convert-linalg-to-loops -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-igelu-lower:
	@${BUDDY_OPT} ./tile-conv-igelu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-igelu-translate:
	@${BUDDY_OPT} ./tile-conv-igelu.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-conv-igelu-asm:
	@${BUDDY_OPT} ./tile-conv-igelu.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-conv-igelu-run:
	@${BUDDY_OPT} ./tile-conv-igelu.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-softmax-lower:
	@${BUDDY_OPT} ./tile-conv-softmax.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-softmax-translate:
	@${BUDDY_OPT} ./tile-conv-softmax.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-conv-softmax-asm:
	@${BUDDY_OPT} ./tile-conv-softmax.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-conv-softmax-run:
	@${BUDDY_OPT} ./tile-conv-softmax.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-relu-lower:
	@${BUDDY_OPT} ./tile-conv-relu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-relu-translate:
	@${BUDDY_OPT} ./tile-conv-relu.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-conv-relu-asm:
	@${BUDDY_OPT} ./tile-conv-relu.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-conv-relu-run:
	@${BUDDY_OPT} ./tile-conv-relu.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-layernorm-lower:
	@${BUDDY_OPT} ./tile-conv-layernorm.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-layernorm-translate:
	@${BUDDY_OPT} ./tile-conv-layernorm.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-conv-layernorm-asm:
	@${BUDDY_OPT} ./tile-conv-layernorm.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-conv-layernorm-run:
	@${BUDDY_OPT} ./tile-conv-layernorm.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-rect-conv-lower:
	@${BUDDY_OPT} ./tile-rect-conv.mlir \
		-lower-gemmini \
		-o log.mlir

tile-rect-conv-translate:
	@${BUDDY_OPT} ./tile-rect-conv.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

tile-rect-conv-asm:
	@${BUDDY_OPT} ./tile-rect-conv.mlir \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

tile-rect-conv-run:
	@${BUDDY_OPT} ./tile-rect-conv.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-matmul-lower:
	@${BUDDY_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-matmul-translate:
	@${BUDDY_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-linalg-matmul-asm:
	@${BUDDY_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-matmul-run:
	@${BUDDY_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nchw-fchw-i8-lower:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-conv2d-nchw-fchw-i8-translate:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nchw-fchw-i8-asm:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nchw-fchw-i8-run:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-dw-conv2d-nchw-chw-i8-lower:
	@${BUDDY_OPT} ./dw_conv_2d_nchw_chw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-dw-conv2d-nchw-chw-i8-translate:
	@${BUDDY_OPT} ./dw_conv_2d_nchw_chw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-linalg-dw-conv2d-nchw-chw-i8-asm:
	@${BUDDY_OPT} ./dw_conv_2d_nchw_chw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-dw-conv2d-nchw-chw-i8-run:
	@${BUDDY_OPT} ./dw_conv_2d_nchw_chw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nchw-fchw-f32-lower:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" \
		-o log.mlir

gemmini-linalg-conv2d-nchw-fchw-f32-translate:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nchw-fchw-f32-asm:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nchw-fchw-f32-run:
	@${BUDDY_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nhwc-hwcf-i8-lower:
	@${BUDDY_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-conv2d-nhwc-hwcf-i8-translate:
	@${BUDDY_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nhwc-hwcf-i8-asm:
	@${BUDDY_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nhwc-hwcf-i8-run:
	@${BUDDY_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nhwc-hwcf-f32-lower:
	@${BUDDY_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" \
		-o log.mlir

gemmini-linalg-conv2d-nhwc-hwcf-f32-translate:
	@${BUDDY_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nhwc-hwcf-f32-asm:
	@${BUDDY_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nhwc-hwcf-f32-run:
	@${BUDDY_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-batch-matmul-lower:
	@${BUDDY_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-batch-matmul-translate:
	@${BUDDY_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-linalg-batch-matmul-asm:
	@${BUDDY_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-batch-matmul-run:
	@${BUDDY_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

linalg-matmul-32x32-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
		-llvm-request-c-wrappers \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops  \
		-lower-gemmini \
		-o log.mlir

linalg-matmul-32x32-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
		-llvm-request-c-wrappers \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops  \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-32x32-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
		-llvm-request-c-wrappers \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops  \
		-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-32x32-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=1 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out 

c-matmul-32x32-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=1 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-matmul-32x32-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-32x32-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-32x32-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-32x32-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=1 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out

gemmini-matmul-32x32-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-32x32-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-matmul-32x32-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-32x32-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=1 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

linalg-matmul-64x64-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-64x64-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-64x64-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-64x64-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=2 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out 

c-matmul-64x64-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=2 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-matmul-64x64-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-64x64-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-64x64-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-64x64-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=2 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out

gemmini-matmul-64x64-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-64x64-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-matmul-64x64-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-64x64-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=2 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 
	
linalg-matmul-128x128-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-128x128-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-128x128-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-128x128-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=3 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out 

c-matmul-128x128-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=3 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-matmul-128x128-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-128x128-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-128x128-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-128x128-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=3 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out

gemmini-matmul-128x128-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-128x128-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-matmul-128x128-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-128x128-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=3 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

linalg-matmul-256x256-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-256x256-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-256x256-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-256x256-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=4 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out 

c-matmul-256x256-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=4 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-matmul-256x256-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-256x256-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-256x256-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-256x256-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=4 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out

gemmini-matmul-256x256-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-256x256-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-matmul-256x256-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-256x256-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=4 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

linalg-matmul-512x512-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-512x512-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-512x512-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-512x512-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=5 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out 

c-matmul-512x512-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=5 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-matmul-512x512-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-512x512-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-512x512-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-512x512-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=5 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out

gemmini-matmul-512x512-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-512x512-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll
	
gemmini-matmul-512x512-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-512x512-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=5 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

linalg-matmul-1024x1024-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-1024x1024-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-1024x1024-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-1024x1024-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=6 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out 

c-matmul-1024x1024-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=6 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-matmul-1024x1024-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-1024x1024-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-1024x1024-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-matmul-1024x1024-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=6 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES} 
	@spike --extension=gemmini pk a.out

gemmini-matmul-1024x1024-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-1024x1024-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-matmul-1024x1024-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-1024x1024-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=6 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

linalg-conv-3x3-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-3x3-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-3x3-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-3x3-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=1 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

c-conv-3x3-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=1 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-conv-3x3-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-3x3-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-3x3-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-3x3-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=1 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-3x3-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-3x3-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-conv-3x3-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-conv-3x3-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=1 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

linalg-conv-5x5-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-5x5-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-5x5-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-5x5-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=2 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

c-conv-5x5-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=2 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-conv-5x5-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-5x5-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-5x5-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-5x5-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=2 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-5x5-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-5x5-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-conv-5x5-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-conv-5x5-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=2 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

linalg-conv-7x7-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-7x7-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-7x7-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-7x7-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=3 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

c-conv-7x7-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=3 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-conv-7x7-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-7x7-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-7x7-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-7x7-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=3 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-7x7-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-7x7-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-conv-7x7-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-conv-7x7-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=3 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

linalg-conv-9x9-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-9x9-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-9x9-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-9x9-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=4 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

c-conv-9x9-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=4 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-conv-9x9-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-9x9-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-9x9-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-9x9-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=4 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-9x9-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-9x9-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-conv-9x9-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-conv-9x9-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=4 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

linalg-conv-11x11-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-11x11-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-11x11-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-11x11-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=5 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

c-conv-11x11-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=5 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-conv-11x11-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-11x11-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-11x11-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-11x11-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=5 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-11x11-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-11x11-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll
	
gemmini-conv-11x11-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-conv-11x11-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=5 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 


linalg-conv-13x13-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-13x13-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-13x13-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-13x13-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=6 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 

c-conv-13x13-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=6 -O2 -static 
	@spike --extension=gemmini pk a.out 

linalg-conv-13x13-cpu-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-13x13-cpu-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-conv-13x13-cpu-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

linalg-conv-13x13-cpu-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=6 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-13x13-gemmini-lower:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-13x13-gemmini-translate:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

gemmini-conv-13x13-gemmini-asm:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.s

gemmini-conv-13x13-gemmini-run:
	@${BUDDY_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o 
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=6 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out 
