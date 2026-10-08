# Runtime/tool selection only. Triton-generated IR is built by triton/build.py.
.DEFAULT_GOAL := runtime
MODEL_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(MODEL_ROOT)/../../../..)
COMMON := $(abspath $(MODEL_ROOT)/../../common)
include $(COMMON)/toolchain.mk
BUDDY_BIN ?= $(REPO_ROOT)/build/bin
BUDDY_OPT ?= $(BUDDY_BIN)/buddy-opt
BUDDY_TRANSLATE ?= $(BUDDY_BIN)/buddy-translate
HOST_CC ?= $(call fpga_llvm_tool,clang)
LLC ?= $(call fpga_llvm_tool,llc)
PYTHON ?= python3
NR_BUILD_DIR := $(MODEL_ROOT)/triton/build/runtime
include $(COMMON)/nr/nr.mk
# Vectorize independent output channels. The K-vectorized decode pass emits
# vfred[uo]sum, whose zero handling fails on the delivered patch6 FPGA.
LINEAR_FP32_PASS ?= --matmul-vectorization='vector-size=16 vector-type=fixed'
LINEAR_RVV_FLAGS ?= -mattr=+m,+a,+f,+d,+c,+v,+zvl512b -riscv-v-vector-bits-min=512 -disable-machine-licm -disable-machine-cse
# Same fixed-vector matmul pass used by Qwen's FP32 attention tiles.
PWCONV_FP32_PASS ?= --matmul-vectorization='vector-size=16 vector-type=fixed'
PWCONV_RVV_FLAGS ?= $(LINEAR_RVV_FLAGS)
STEM_FP32_PASS ?= $(PWCONV_FP32_PASS)
STEM_RVV_FLAGS ?= $(PWCONV_RVV_FLAGS)
CONFIG_VARIABLES := BUDDY_OPT BUDDY_TRANSLATE HOST_CC LLC RISCV_CC RISCV_LD RISCV_OBJCOPY RISCV_OBJDUMP NR_CFLAGS NR_OBJECTS NR_LINKER_SCRIPT LINEAR_FP32_PASS LINEAR_RVV_FLAGS PWCONV_FP32_PASS PWCONV_RVV_FLAGS STEM_FP32_PASS STEM_RVV_FLAGS
$(foreach variable,$(CONFIG_VARIABLES),$(eval export MOBILENET_CFG_$(variable) := $($(variable))))
.PHONY: runtime print-config
runtime: $(NR_OBJECTS)
print-config:
	@$(PYTHON) $(MODEL_ROOT)/triton/build.py --print-config
