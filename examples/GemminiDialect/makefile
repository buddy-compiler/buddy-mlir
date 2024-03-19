#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc 
INTERFACES:= ../../frontend/Interfaces

mvin-mvout-run:
	@${BUDDY_OPT} ./mvin-mvout.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matrix-add-run:
	@${BUDDY_OPT} ./matrix-add.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matrix-add-scale-run:
	@${BUDDY_OPT} ./matrix-add-scale.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

transpose-run:
	@${BUDDY_OPT} ./transpose.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matmul-os-run:
	@${BUDDY_OPT} ./matmul-os.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

compute-accumulated-run:
	@${BUDDY_OPT} ./compute-accumulated.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matmul-ws-run:
	@${BUDDY_OPT} ./matmul-ws.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-run:
	@${BUDDY_OPT} ./tile-matmul.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-os-run:
	@${BUDDY_OPT} ./tile-matmul-os.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-igelu-run:
	@${BUDDY_OPT} ./tile-matmul-ws-igelu.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-relu-run:
	@${BUDDY_OPT} ./tile-matmul-ws-relu.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-softmax-run:
	@${BUDDY_OPT} ./tile-matmul-ws-softmax.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-layernorm-run:
	@${BUDDY_OPT} ./tile-matmul-ws-layernorm.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-run:
	@${BUDDY_OPT} ./tile-conv.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-igelu-run:
	@${BUDDY_OPT} ./tile-conv-igelu.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-softmax-run:
	@${BUDDY_OPT} ./tile-conv-softmax.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-relu-run:
	@${BUDDY_OPT} ./tile-conv-relu.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-layernorm-run:
	@${BUDDY_OPT} ./tile-conv-layernorm.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-rect-conv-run:
	@${BUDDY_OPT} ./tile-rect-conv.mlir -lower-gemmini | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

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
