#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
LLI := ../../llvm/build/bin/lli
LLC := ../../llvm/build/bin/llc
OPT := ../../llvm/build/bin/opt
OPT_FLAG := -O3
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

RISCV_GNU_TOOLCHAIN := ../../thirdparty/build-riscv-gnu-toolchain
RISCV_GNU_TOOLCHAIN_SYSROOT := ../../thirdparty/build-riscv-gnu-toolchain/sysroot
QEMU := ../../thirdparty/qemu/build/riscv64-linux-user/qemu-riscv64
LOCAL_CLANG := ../../thirdparty/build-local-clang/bin/clang
CROSS_LLI := ../../thirdparty/build-cross-clang/bin/lli
CROSS_MLIR_CPU_RUNNER := ../../thirdparty/build-cross-mlir/bin/mlir-cpu-runner
CROSS_MLIR_C_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_c_runner_utils.so
CROSS_MLIR_RUNNER_UTILS := ../../thirdparty/build-cross-mlir/lib/libmlir_runner_utils.so
CROSS_MLIR_LIB := ../../thirdparty/build-cross-mlir/lib

resnet-18-lower:
	@${MLIR_OPT} ./ResNet-18.mlir \
		-pass-pipeline="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-tensor),func.func(tosa-to-linalg),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		--test-linalg-transform-patterns="test-generalize-pad-tensor" \
		--linalg-bufferize \
		--convert-linalg-to-loops \
		--func-bufferize \
		--arith-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--convert-memref-to-llvm \
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts \
		-o ./log.mlir

# Recommand to test OpenCV on RISC-V before moving on model experiments.
# To get things ready, please first cross-compile OpenCV on RISC-V and make sure PKG_CONFIG_PATH and LD_LIBRARY_PATH are correctly set as follows: \
$ git clone git@github.com:opencv/opencv.git \
$ cd opencv \
$ mkdir build && cd build \
$ cmake -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/riscv64-gcc.toolchain.cmake -DOPENCV_GENERATE_PKGCONFIG=ON ../ \
$ make -j$(nproc) \
$ make install \
$ sudo find / -name opencv4.pc # Mark the output as <path/to/opencv/lib>/pkgconfig/opencv4.pc \
$ export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:<path/to/opencv/lib>/pkgconfig \
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path/to/opencv/lib>
test-opencv-run-rv:
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ -c -I ../../frontend/Interfaces/ `pkg-config opencv4 --cflags` TestOpenCV.cpp `pkg-config opencv4 --libs`
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ `pkg-config opencv4 --cflags` TestOpenCV.o `pkg-config opencv4 --libs`\
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -no-pie -mabi=lp64d -o TestOpenCV
	@LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 TestOpenCV

# Run end-to-end ResNet18 benchmark on RISC-V. Execution in QEMU simulator may take a while.
resnet-18-run-rv: resnet-18-default.o
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ -c -I ../../frontend/Interfaces/ `pkg-config opencv4 --cflags` ResNet18DefaultBenchmark.cpp `pkg-config opencv4 --libs`
	@${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ `pkg-config opencv4 --cflags` resnet-18-default.o ResNet18DefaultBenchmark.o `pkg-config opencv4 --libs`\
		-L${CROSS_MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -no-pie -mabi=lp64d -fPIC -o ResNet18Default
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):${CROSS_MLIR_LIB} ${QEMU} -L ${RISCV_GNU_TOOLCHAIN_SYSROOT} -cpu rv64,x-v=true,vlen=128 ResNet18Default

resnet-18-default.o:
	@wget https://github.com/buddy-compiler/buddy-benchmark/raw/main/benchmarks/DeepLearning/Models/ResNet-18/ResNet-18.mlir -O ResNet-18-weighted.mlir
	@${MLIR_OPT} ./ResNet-18-weighted.mlir \
		-pass-pipeline="builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-tensor),func.func(tosa-to-linalg),func.func(tosa-to-arith))" | \
	${MLIR_OPT} \
		--test-linalg-transform-patterns="test-generalize-pad-tensor" \
		--linalg-bufferize \
		--convert-linalg-to-loops \
		--func-bufferize \
		--arith-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		--convert-vector-to-scf \
		--convert-scf-to-cf \
		--expand-strided-metadata \
		--lower-affine \
		--convert-vector-to-llvm \
		--memref-expand \
		--arith-expand \
		--convert-arith-to-llvm \
		--convert-memref-to-llvm \
		--convert-math-to-llvm \
		--llvm-request-c-wrappers \
		--convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${MLIR_TRANSLATE} --mlir-to-llvmir | \
	${LLC} -O3 -mtriple riscv64 -target-abi lp64d \
		-mattr=+m,+d,+v --filetype=obj -riscv-v-vector-bits-min=128 \
		-o ./resnet-18-default.o

.PHONY: clean
clean:
	rm TestOpenCV.o TestOpenCV resnet-18-default.o ResNet18DefaultBenchmark.o ResNet18Default ResNet-18-weighted.mlir
