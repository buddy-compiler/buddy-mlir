.DEFAULT_GOAL := host

BUILD_TYPE ?= Release
PYTHON ?= $(if $(wildcard $(CURDIR)/.venv/bin/python),$(CURDIR)/.venv/bin/python,python3)
PARALLEL ?=
# Linker name accepted by the selected compiler, for example `ld`, `lld`,
# `mold`, or `gold`. Leave empty for the compiler default.
LINKER ?=
# Compiler commands, optionally overridden for a different toolchain.
CC ?= cc
CXX ?= c++

# GCC/Clang spell GNU ld as `bfd` for -fuse-ld. Keep `LINKER=ld` convenient.
LINKER_NAME := $(if $(filter ld ld.bfd,$(LINKER)),bfd,$(LINKER))
LINKER_FLAG := $(if $(strip $(LINKER_NAME)),-fuse-ld=$(LINKER_NAME),)

LLVM_BUILD ?= $(CURDIR)/llvm/build
BUDDY_BUILD ?= $(CURDIR)/build
PREFIX ?= $(CURDIR)/install

# RISC-V GNU/Linux sysroot and toolchain (built from the repository submodule).
RISCV_SOURCE ?= $(CURDIR)/thirdparty/riscv-gnu-toolchain
RISCV_BUILD ?= $(RISCV_SOURCE)/build
RISCV_INSTALL ?= $(CURDIR)/thirdparty/riscv
RISCV_ARCH ?= rv64gcv
RISCV_ABI ?= lp64d
RISCV_MLIR_BUILD ?= $(CURDIR)/llvm/build-cross-mlir-rv
RISCV_OMP_BUILD ?= $(CURDIR)/llvm/build-omp-shared-rv
RISCV_TARGET := riscv64-unknown-linux-gnu
RISCV_TARGET_FLAGS := --target=$(RISCV_TARGET) --sysroot=$(RISCV_INSTALL)/sysroot --gcc-toolchain=$(RISCV_INSTALL)

# Model targets are generated from the checked-in spec files. Keep their
# CMake cache separate from the host build so a cross-model build cannot alter
# the compiler settings used by `make host`.
RISCV_MODEL_BUILD ?= $(CURDIR)/build-riscv
RISCV_MODEL_LOCAL ?=
RISCV_MODEL_HF_CONFIG ?=
RISCV_MODEL_DEFAULT ?= deepseek_r1-f32
riscv_model_family = $(notdir $(patsubst %/specs/,%,$(dir $(1))))
riscv_model_variant = $(basename $(notdir $(1)))
riscv_model_target = riscv-model-$(call riscv_model_family,$(1))-$(call riscv_model_variant,$(1))
riscv_model_build = $(RISCV_MODEL_BUILD)/$(call riscv_model_family,$(1))-$(call riscv_model_variant,$(1))
RISCV_MODEL_SPECS := $(sort $(wildcard $(CURDIR)/models/*/specs/*.json))
RISCV_MODEL_TARGETS := $(foreach _spec,$(RISCV_MODEL_SPECS),$(call riscv_model_target,$(_spec)))
RISCV_MODEL_FAMILIES := $(sort $(foreach _spec,$(RISCV_MODEL_SPECS),$(call riscv_model_family,$(_spec))))
# Family aliases prefer f32, then base, then the first checked-in spec.
riscv_model_specs_for = $(filter $(CURDIR)/models/$(1)/specs/%.json,$(RISCV_MODEL_SPECS))
riscv_model_default_spec = $(firstword $(wildcard $(CURDIR)/models/$(1)/specs/f32.json) $(wildcard $(CURDIR)/models/$(1)/specs/base.json) $(call riscv_model_specs_for,$(1)))
riscv_model_default_target = $(call riscv_model_target,$(call riscv_model_default_spec,$(1)))
RISCV_MODEL_CMAKE_ARGS := \
	--cmake-args=-DLLVM_DIR=$(LLVM_BUILD)/lib/cmake/llvm \
	--cmake-args=-DMLIR_DIR=$(LLVM_BUILD)/lib/cmake/mlir \
	--cmake-args=-DPython3_EXECUTABLE=$(PYTHON) \
	--cmake-args=-DPython_EXECUTABLE=$(PYTHON) \
	--cmake-args=-DBUDDY_ENABLE_TESTS=OFF \
	--cmake-args=-DBUDDY_PACKAGE_VERSION=$(BUDDY_PACKAGE_VERSION)
RISCV_MODEL_OPTIONAL_ARGS := \
	$(if $(strip $(RISCV_MODEL_LOCAL)),--local-model "$(RISCV_MODEL_LOCAL)",) \
	$(if $(strip $(RISCV_MODEL_HF_CONFIG)),--hf-config "$(RISCV_MODEL_HF_CONFIG)",)

LLVM_CMAKE_ARGS := \
	-DLLVM_ENABLE_PROJECTS=mlir\;clang \
	-DLLVM_ENABLE_RUNTIMES=openmp \
	-DLLVM_TARGETS_TO_BUILD=host\;RISCV \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DOPENMP_ENABLE_LIBOMPTARGET=OFF \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
	-DCMAKE_C_COMPILER=$(CC) \
	-DCMAKE_CXX_COMPILER=$(CXX) \
	-DPython3_EXECUTABLE=$(PYTHON) \
	-DPython_EXECUTABLE=$(PYTHON) \
	-DLLVM_USE_LINKER=$(LINKER_NAME)

BUDDY_CMAKE_ARGS := \
	-DMLIR_DIR=$(LLVM_BUILD)/lib/cmake/mlir \
	-DLLVM_DIR=$(LLVM_BUILD)/lib/cmake/llvm \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
	-DCMAKE_C_COMPILER=$(CC) \
	-DCMAKE_CXX_COMPILER=$(CXX) \
	-DPython3_EXECUTABLE=$(PYTHON) \
	-DPython_EXECUTABLE=$(PYTHON) \
	-DCMAKE_EXE_LINKER_FLAGS=$(LINKER_FLAG) \
	-DCMAKE_SHARED_LINKER_FLAGS=$(LINKER_FLAG) \
	-DCMAKE_MODULE_LINKER_FLAGS=$(LINKER_FLAG)

.PHONY: help list-riscv-models
help:
	@printf '%s\n' \
		'Host:' \
		'  make host                 Configure and build host LLVM/Buddy' \
		'  make install              Install the host build (PREFIX=install)' \
		'' \
		'RISC-V:' \
		'  make riscv                Build the RISC-V toolchain and runtimes' \
		'  make riscv-deps           Alias for make riscv' \
		'  make install-riscv        Build/install RISC-V dependencies' \
		'  make riscv-model          Build RISCV_MODEL_DEFAULT=$(RISCV_MODEL_DEFAULT)' \
		'  make riscv-model-<name>   Build a family default or one spec variant' \
		'' \
		'Cleanup:' \
		'  make clean-models         Remove generated model artifacts' \
		'  make clean-host           Remove host LLVM/Buddy build trees' \
		'  make clean-riscv          Remove RISC-V build/install trees' \
		'  make clean-all            Remove host and RISC-V build trees' \
		'' \
		'Available RISC-V model targets:'
	@$(MAKE) --no-print-directory list-riscv-models

list-riscv-models:
	@printf '  %s\n' $(addprefix riscv-model-,$(RISCV_MODEL_FAMILIES))
	@printf '  %s\n' $(RISCV_MODEL_TARGETS)

.PHONY: host
host:
	cmake -G Ninja -S llvm/llvm -B $(LLVM_BUILD) $(LLVM_CMAKE_ARGS)
	cmake --build $(LLVM_BUILD) --parallel $(PARALLEL)
	cmake -G Ninja -S . -B $(BUDDY_BUILD) $(BUDDY_CMAKE_ARGS)
	cmake --build $(BUDDY_BUILD) --parallel $(PARALLEL)

.PHONY: install install-riscv
install: host
	cmake --install "$(BUDDY_BUILD)" --prefix "$(PREFIX)"

install-riscv: riscv-deps
	@printf 'RISC-V toolchain and runtimes are installed in %s\n' "$(RISCV_INSTALL)"

.PHONY: clean-models clean-host clean-riscv clean-all clean
clean-models:
	rm -rf -- "$(BUDDY_BUILD)/models"

clean-host:
	rm -rf -- "$(LLVM_BUILD)" "$(BUDDY_BUILD)"

clean-riscv:
	rm -rf -- \
		"$(RISCV_BUILD)" \
		"$(RISCV_INSTALL)" \
		"$(RISCV_MLIR_BUILD)" \
		"$(RISCV_OMP_BUILD)" \
		"$(RISCV_MODEL_BUILD)"

clean-all: clean-host clean-riscv

clean: clean-host

.PHONY: riscv
riscv: host
	git submodule update --init --recursive thirdparty/riscv-gnu-toolchain
	mkdir -p "$(RISCV_BUILD)"
	@if test ! -f "$(RISCV_BUILD)/Makefile"; then \
		cd "$(RISCV_BUILD)" && \
		"$(RISCV_SOURCE)/configure" \
			--prefix="$(RISCV_INSTALL)" \
			--with-arch="$(RISCV_ARCH)" \
			--with-abi="$(RISCV_ABI)" \
			--disable-multilib \
			--disable-llvm \
			--disable-gdb; \
	fi
	make -C "$(RISCV_BUILD)" linux -j MAKEINFO=:

	cmake -G Ninja -S llvm/llvm -B "$(RISCV_MLIR_BUILD)" \
		-DLLVM_ENABLE_PROJECTS=mlir \
		-DLLVM_BUILD_EXAMPLES=OFF \
		-DLLVM_INCLUDE_TESTS=OFF \
		-DLLVM_INCLUDE_DOCS=OFF \
		-DMLIR_INCLUDE_TESTS=OFF \
		-DMLIR_ENABLE_EXECUTION_ENGINE=ON \
		-DLLVM_ENABLE_PIC=ON \
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_CROSSCOMPILING=True \
		-DLLVM_TARGET_ARCH=RISCV64 \
		-DLLVM_TARGETS_TO_BUILD=RISCV \
		-DLLVM_NATIVE_ARCH=RISCV \
		-DLLVM_HOST_TRIPLE=$(RISCV_TARGET) \
		-DLLVM_DEFAULT_TARGET_TRIPLE=$(RISCV_TARGET) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX="$(RISCV_INSTALL)" \
		-DCMAKE_C_COMPILER="$(LLVM_BUILD)/bin/clang" \
		-DCMAKE_CXX_COMPILER="$(LLVM_BUILD)/bin/clang++" \
		-DCMAKE_C_FLAGS="$(RISCV_TARGET_FLAGS)" \
		-DCMAKE_CXX_FLAGS="$(RISCV_TARGET_FLAGS)" \
		-DLLVM_NATIVE_TOOL_DIR="$(LLVM_BUILD)/bin" \
		-DLLVM_ENABLE_ZSTD=OFF
	cmake --build "$(RISCV_MLIR_BUILD)" --target mlir_c_runner_utils --parallel
	cmake --install "$(RISCV_MLIR_BUILD)" --component mlir_float16_utils
	cmake --install "$(RISCV_MLIR_BUILD)" --component mlir_apfloat_wrappers
	cmake --install "$(RISCV_MLIR_BUILD)" --component mlir_c_runner_utils

	# OpenMP is a standalone runtime build; the explicit Clang compiler below is
	# sufficient when runtime tests are disabled.
	cmake -G Ninja -S llvm/runtimes -B "$(RISCV_OMP_BUILD)" \
		-DLLVM_ENABLE_RUNTIMES=openmp \
		-DLLVM_DEFAULT_TARGET_TRIPLE=$(RISCV_TARGET) \
		-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF \
		-DLLVM_INCLUDE_TESTS=OFF \
		-DLLVM_INCLUDE_DOCS=OFF \
		-DOPENMP_ENABLE_LIBOMPTARGET=OFF \
		-DOPENMP_ENABLE_OMPT_TOOLS=OFF \
		-DLIBOMP_ENABLE_SHARED=ON \
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX="$(RISCV_INSTALL)" \
		-DCMAKE_C_COMPILER="$(LLVM_BUILD)/bin/clang" \
		-DCMAKE_CXX_COMPILER="$(LLVM_BUILD)/bin/clang++" \
		-DCMAKE_C_FLAGS="$(RISCV_TARGET_FLAGS)" \
		-DCMAKE_CXX_FLAGS="$(RISCV_TARGET_FLAGS)" \
		-DCMAKE_ASM_FLAGS="$(RISCV_TARGET_FLAGS)"
	cmake --build "$(RISCV_OMP_BUILD)" --target omp --parallel
	cmake --install "$(RISCV_OMP_BUILD)" --component openmp

.PHONY: riscv-deps
riscv-deps: riscv

.PHONY: riscv-model
riscv-model: riscv-model-$(RISCV_MODEL_DEFAULT)

define RISCV_MODEL_FAMILY_template
.PHONY: riscv-model-$(1)
riscv-model-$(1): $(call riscv_model_default_target,$(1))
endef
$(foreach _family,$(RISCV_MODEL_FAMILIES),$(eval $(call RISCV_MODEL_FAMILY_template,$(_family))))

define RISCV_MODEL_template
.PHONY: $(call riscv_model_target,$(1))
$(call riscv_model_target,$(1)): riscv
	cmake --build "$(BUDDY_BUILD)" --target python-package-buddy-mlir --parallel $(PARALLEL)
	$(PYTHON) "$(CURDIR)/tools/buddy-codegen/build_model.py" \
		--spec "$(1)" \
		--build-dir "$(call riscv_model_build,$(1))" \
		--is-rvv-crosscompile \
		--riscv-gnu-toolchain "$(RISCV_INSTALL)" \
		--riscv-omp-shared "$(RISCV_INSTALL)/lib/libomp.so" \
		--riscv-mlir-c-runner-utils "$(RISCV_INSTALL)/lib/libmlir_c_runner_utils.so" \
		--buddy-mlir-build-dir "$(BUDDY_BUILD)" \
		$(RISCV_MODEL_OPTIONAL_ARGS) \
		$(RISCV_MODEL_CMAKE_ARGS) \
		$(if $(strip $(PARALLEL)),--jobs $(PARALLEL),)
endef
$(foreach _spec,$(RISCV_MODEL_SPECS),$(eval $(call RISCV_MODEL_template,$(_spec))))
