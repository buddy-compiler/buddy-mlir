# Link already compiled Triton kernel/adapter pairs with the common NR runtime.
.DEFAULT_GOAL := all
OPTIMIZATION := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MODEL := $(abspath $(OPTIMIZATION)/..)
QWEN := $(abspath $(MODEL)/..)
REPO_ROOT := $(abspath $(MODEL)/../../../..)
COMMON := $(abspath $(MODEL)/../../common)
include $(COMMON)/toolchain.mk
NR_BUILD_DIR := $(BUILD)/runtime
include $(COMMON)/nr/nr.mk
BENCHMARK ?= dequant
HARNESS_DEFINES ?= -DDQ_ROWS=$(ROWS) -DDQ_COLS=$(COLS) -DDQ_REPEATS=$(REPEATS)
STEM := $(BENCHMARK)-ab

OBJECTS := $(BUILD)/baseline.kernel.o $(BUILD)/baseline.adapter.o \
  $(BUILD)/optimized.kernel.o $(BUILD)/optimized.adapter.o \
  $(BUILD)/launch.o $(NR_OBJECTS)
.PHONY: all
all: $(BUILD)/$(STEM).bin

$(BUILD)/launch.o: $(OPTIMIZATION)/$(BENCHMARK)_ab_launch.c $(QWEN)/support.h $(MAKEFILE_LIST)
	$(RISCV_CC) $(NR_CFLAGS) -Wall -Wextra -Werror -I$(QWEN) \
	  $(HARNESS_DEFINES) -c $< -o $@

$(BUILD)/$(STEM).elf: $(OBJECTS) $(NR_LINKER_SCRIPT) $(MAKEFILE_LIST)
	$(RISCV_LD) -T $(NR_LINKER_SCRIPT) --gc-sections -Map=$(BUILD)/$(STEM).map -o $@ $(OBJECTS)

$(BUILD)/$(STEM).bin: $(BUILD)/$(STEM).elf
	$(PYTHON) $(COMMON)/../tools/check_nr_elf.py $< --objdump $(RISCV_OBJDUMP) --output $(BUILD)/elf-audit.json
	$(RISCV_OBJCOPY) -O binary $< $@
