# ===- toolchain.mk -----------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Include from an example one directory below BOSCAMEFPGA.
# Prefer the repository LLVM build, then fall back to tools on PATH.
# All values can be overridden with make NAME=value or environment variables.
#
# ===---------------------------------------------------------------------------
REPO_ROOT ?= ../../..
LLVM_BIN ?= $(if $(wildcard $(REPO_ROOT)/llvm/build/bin/clang),$(REPO_ROOT)/llvm/build/bin)
fpga_llvm_tool = $(if $(and $(LLVM_BIN),$(wildcard $(LLVM_BIN)/$(1))),$(LLVM_BIN)/$(1),$(1))

RISCV_CC ?= $(call fpga_llvm_tool,clang) --target=riscv64-unknown-elf
RISCV_LD ?= $(call fpga_llvm_tool,ld.lld) -m elf64lriscv
RISCV_OBJCOPY ?= $(call fpga_llvm_tool,llvm-objcopy)
RISCV_OBJDUMP ?= $(call fpga_llvm_tool,llvm-objdump)
