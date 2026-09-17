# Shared build for one kernel.mlir + launch.c operator directory.
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
ROOT := ..
COMMON := ../../common
REPO_ROOT := ../../../..
include $(COMMON)/toolchain.mk
BUDDY_BIN ?= $(firstword $(foreach dir,$(REPO_ROOT)/build-migrate/bin $(REPO_ROOT)/build/bin,$(if $(wildcard $(dir)/buddy-opt),$(dir))))
BUDDY_OPT ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-opt,buddy-opt)
BUDDY_TRANSLATE ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-translate,buddy-translate)
LLC ?= $(call fpga_llvm_tool,llc)
HOST_CC ?= $(call fpga_llvm_tool,clang)
PYTHON ?= python3
NAME := $(notdir $(CURDIR))
BUILD := build
TOOLS := ../../tools
NR := $(COMMON)/nr
AME_PASS ?= --lower-linalg-to-boscame=target=nr-fpga
# Opt-in v0.5 operand encoding; keep the previously verified wrappers by default.
# Export in print-config/config.json so changing this invalidates built kernels.
AME_GPR_MODE ?= fixed
ifneq ($(AME_GPR_MODE),fixed)
ifneq ($(AME_GPR_MODE),direct)
$(error AME_GPR_MODE must be fixed or direct)
endif
endif
# Experimental adjacent identical fence sharing; retains the ELF fence audit.
NR_COALESCE_FENCES ?= 0
ifneq ($(NR_COALESCE_FENCES),0)
ifneq ($(NR_COALESCE_FENCES),1)
$(error NR_COALESCE_FENCES must be 0 or 1)
endif
endif
FP32_PASS ?= --matmul-transpose-b-vectorization-decode='vector-size=16 unroll=1 n-tile=1'
MATMUL_FP32_PASS ?= --matmul-vectorization='vector-size=16 vector-type=fixed'
BATCH_FP32_PASS ?= --batchmatmul-optimize=vector-size=16
RVV_FLAGS ?= -mattr=+m,+a,+f,+d,+c,+v,+zvl512b,+xboscame \
 -riscv-v-vector-bits-min=512
# Transpose-B dot reductions otherwise acquire an unsupported whole-register
# COPY for the zero accumulator. Attention uses its separately board-validated
# default machine optimizations and does not generate that COPY pattern.
RVV_LINEAR_FLAGS ?= $(RVV_FLAGS) -disable-machine-licm -disable-machine-cse
SCALAR_FLAGS ?= -mattr=+m,+a,+f,+d,+c,-v,+xboscame
RVV_ENABLED := $(if $(filter matmul_%_f32 attention_qk_% attention_pv_%,$(NAME)),1,0)
KERNEL_LLFLAGS := -O2 -filetype=asm -mtriple=riscv64 -target-abi=lp64d \
 $(if $(filter 1,$(RVV_ENABLED)),$(if $(filter matmul_%_f32,$(NAME)),$(RVV_LINEAR_FLAGS),$(RVV_FLAGS)),$(SCALAR_FLAGS)) -code-model=medium
LOWER := --convert-linalg-to-loops --expand-strided-metadata --lower-affine \
 --convert-vector-to-scf --convert-vector-to-llvm \
 --convert-math-to-llvm --convert-math-to-libm --convert-scf-to-cf \
 --convert-cf-to-llvm --convert-arith-to-llvm --convert-index-to-llvm \
 --memref-expand --finalize-memref-to-llvm --convert-func-to-llvm \
 --reconcile-unrealized-casts
CFLAGS := -march=rv64gc_zicbom -mabi=lp64d -mcmodel=medany -O2 -nostdlib \
 -ffreestanding -fno-builtin -fno-pie -fno-vectorize -fno-slp-vectorize \
 -ffp-contract=off -ffunction-sections -fdata-sections -I$(ROOT) -I$(NR) -I$(COMMON)/uart
HOST_CFLAGS := -O2 -ffp-contract=off -DHOST_TEST -I$(ROOT)
RUNTIME_OBJS := $(addprefix $(BUILD)/,crt.o nr_runtime.o nr_math.o ame_sync.o nr_copy.o support.o)
CONFIG_HELPER := $(ROOT)/tools/build_suite.py
CONFIG_STAMP := $(BUILD)/config.json
# Export configuration as data, without interpolating tool commands into Python
# or shell string literals. Suite builds consume exactly this configuration.
CONFIG_VARIABLES := BUDDY_BIN BUDDY_OPT BUDDY_TRANSLATE LLVM_BIN LLC HOST_CC \
 PYTHON RISCV_CC RISCV_LD RISCV_OBJCOPY RISCV_OBJDUMP CFLAGS HOST_CFLAGS LOWER AME_PASS AME_GPR_MODE NR_COALESCE_FENCES \
 BUILD NR ROOT TOOLS RUNTIME_OBJS FP32_PASS MATMUL_FP32_PASS BATCH_FP32_PASS RVV_FLAGS RVV_LINEAR_FLAGS SCALAR_FLAGS RVV_ENABLED KERNEL_LLFLAGS
$(foreach variable,$(CONFIG_VARIABLES),$(eval export QWEN_CFG_$(variable) := $($(variable))))

.PHONY: all check clean run print-config FORCE
print-config:
	@$(PYTHON) $(CONFIG_HELPER) --print-config
# The helper preserves the timestamp when the effective options and tool
# executable identities are unchanged. Changing AME_PASS, toolchain, or flags
# therefore rebuilds kernels and launchers instead of reusing an older target.
$(CONFIG_STAMP): FORCE $(MAKEFILE_LIST) $(CONFIG_HELPER) | $(BUILD)
	@$(PYTHON) $(CONFIG_HELPER) --config-stamp $@
$(BUILD)/kernel.ame.mlir $(BUILD)/kernel.host.mlir $(BUILD)/kernel.s \
 $(BUILD)/kernel.vector.mlir $(BUILD)/kernel.o $(BUILD)/launch.o $(BUILD)/host-check $(RUNTIME_OBJS): $(CONFIG_STAMP) $(MAKEFILE_LIST)
all: $(BUILD)/$(NAME).bin
$(BUILD):
	mkdir -p $@
$(BUILD)/kernel.ame.mlir: kernel.mlir metadata.json | $(BUILD)
	$(BUDDY_OPT) $< $(AME_PASS) -o $@
	@$(PYTHON) $(CONFIG_HELPER) --audit-lowering $@ metadata.json
$(BUILD)/kernel.vector.mlir: $(BUILD)/kernel.ame.mlir metadata.json $(ROOT)/tools/vectorize_nr.py
	$(PYTHON) $(ROOT)/tools/vectorize_nr.py --input $< --metadata metadata.json --output $@
$(BUILD)/kernel.llvm.mlir: $(BUILD)/kernel.vector.mlir
	$(BUDDY_OPT) $< --lower-bosc-ame $(LOWER) -o $@
$(BUILD)/kernel.raw.ll: $(BUILD)/kernel.llvm.mlir
	$(BUDDY_TRANSLATE) --buddy-to-llvmir $< -o $@
$(BUILD)/kernel.ll: $(BUILD)/kernel.raw.ll metadata.json $(ROOT)/tools/vectorize_nr.py
	$(PYTHON) $(ROOT)/tools/vectorize_nr.py --prepare-llvm --input $< --metadata metadata.json --output $@
$(BUILD)/kernel.s: $(BUILD)/kernel.ll
	$(LLC) $< $(KERNEL_LLFLAGS) -o $@
$(BUILD)/kernel.nr.S: $(BUILD)/kernel.s $(TOOLS)/ame_to_word.py $(TOOLS)/restrict_fpga_assembly.py $(TOOLS)/nr_isa.py
	$(PYTHON) $(TOOLS)/ame_to_word.py --gpr-mode=$(AME_GPR_MODE) < $< > $(BUILD)/kernel.encoded.s
	$(PYTHON) $(TOOLS)/restrict_fpga_assembly.py $(if $(filter 1,$(NR_COALESCE_FENCES)),--coalesce-fences) < $(BUILD)/kernel.encoded.s > $@
$(BUILD)/kernel.o: $(BUILD)/kernel.nr.S
	$(RISCV_CC) $(CFLAGS) $(if $(filter 1,$(RVV_ENABLED)),-march=rv64gcv_zicbom) -c $< -o $@
$(BUILD)/launch.o: launch.c $(ROOT)/support.h $(NR)/nr_runtime.h | $(BUILD)
	$(RISCV_CC) $(CFLAGS) -c $< -o $@
$(BUILD)/support.o: $(ROOT)/support.c $(ROOT)/support.h $(NR)/nr_runtime.h | $(BUILD)
	$(RISCV_CC) $(CFLAGS) -c $< -o $@
$(BUILD)/%.o: $(NR)/%.c $(NR)/nr_runtime.h | $(BUILD)
	$(RISCV_CC) $(CFLAGS) -c $< -o $@
$(BUILD)/nr_copy.o: $(NR)/nr_copy.S | $(BUILD)
	$(RISCV_CC) $(CFLAGS) -march=rv64gcv_zicbom -c $< -o $@
$(BUILD)/crt.o: $(NR)/crt.S | $(BUILD)
	$(RISCV_CC) $(CFLAGS) -c $< -o $@
$(BUILD)/$(NAME).elf: $(BUILD)/kernel.o $(BUILD)/launch.o $(RUNTIME_OBJS) $(NR)/nr.ld
	$(RISCV_LD) --gc-sections -T $(NR)/nr.ld -Map=$(BUILD)/$(NAME).map -o $@ $(filter %.o,$^)
$(BUILD)/elf-audit.json: $(BUILD)/$(NAME).elf $(TOOLS)/check_nr_elf.py $(TOOLS)/nr_isa.py
	$(PYTHON) $(TOOLS)/check_nr_elf.py $< --objdump "$(RISCV_OBJDUMP)" --output $@
$(BUILD)/$(NAME).bin: $(BUILD)/$(NAME).elf $(BUILD)/elf-audit.json
	$(RISCV_OBJCOPY) -O binary $< $@
# CPU control executes the same linalg kernel, lowered to loops, against the
# launcher's independent oracle. It does not replace the FPGA AME test.
$(BUILD)/kernel.host.mlir: kernel.mlir | $(BUILD)
	$(BUDDY_OPT) $< $(LOWER) -o $@
$(BUILD)/kernel.host.ll: $(BUILD)/kernel.host.mlir
	$(BUDDY_TRANSLATE) --buddy-to-llvmir $< -o $@
$(BUILD)/host-check: $(BUILD)/kernel.host.ll launch.c $(ROOT)/support.c $(ROOT)/support.h $(ROOT)/tools/host_main.c
	$(HOST_CC) $(HOST_CFLAGS) $(filter %.c %.ll,$^) -lm -o $@
check: $(BUILD)/host-check
	./$(BUILD)/host-check
ifeq ($(RVV_ENABLED),1)
# Independently check the actual vector-lowered LLVM program as well as the
# original linalg control. Host code generation does not validate FPGA ISA.
$(BUILD)/host-vector-check: $(BUILD)/kernel.ll launch.c $(ROOT)/support.c $(ROOT)/support.h $(ROOT)/tools/host_main.c
	$(HOST_CC) $(HOST_CFLAGS) $(filter %.c %.ll,$^) -lm -o $@
.PHONY: check-vector
check-vector: $(BUILD)/host-vector-check
	./$(BUILD)/host-vector-check
check: check-vector
endif
run: all
	../../fpga_run.sh $(BUILD)/$(NAME).bin --fpga=$(or $(FPGA),5) --capture-seconds=$(or $(CAPTURE_SECONDS),900) --completion-marker='[nr] RA returned:'
clean:
	rm -rf $(BUILD)
