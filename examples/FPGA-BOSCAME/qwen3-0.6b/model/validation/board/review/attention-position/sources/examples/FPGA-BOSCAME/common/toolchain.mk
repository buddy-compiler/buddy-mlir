# Include from an example one directory below FPGA-BOSCAME.
# Prefer the repository LLVM build, then fall back to tools on PATH.
# All values can be overridden with make NAME=value or environment variables.
REPO_ROOT ?= ../../..
LLVM_BIN ?= $(firstword $(foreach dir,$(REPO_ROOT)/llvm/build-2d26/bin $(REPO_ROOT)/llvm/build/bin,$(if $(wildcard $(dir)/clang),$(dir))))
fpga_llvm_tool = $(if $(and $(LLVM_BIN),$(wildcard $(LLVM_BIN)/$(1))),$(LLVM_BIN)/$(1),$(1))

RISCV_CC ?= $(call fpga_llvm_tool,clang) --target=riscv64-unknown-elf
RISCV_LD ?= $(call fpga_llvm_tool,ld.lld) -m elf64lriscv
RISCV_OBJCOPY ?= $(call fpga_llvm_tool,llvm-objcopy)
RISCV_OBJDUMP ?= $(call fpga_llvm_tool,llvm-objdump)
