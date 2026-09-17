# Called by tools/build_tokenizer_probe.py after fixture/blob preparation.
.DEFAULT_GOAL := all
MODEL ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
BUILD ?= $(MODEL)/build/tokenizer-probe
REPO_ROOT := $(abspath $(MODEL)/../../../..)
COMMON := $(MODEL)/../../common
include $(COMMON)/toolchain.mk
NR_BUILD_DIR := $(BUILD)/runtime
include $(COMMON)/nr/nr.mk

TEXT_OBJECTS := $(addprefix $(BUILD)/,tokenizer_resource.o tokenizer_encode.o unicode_tables.o)
OBJECTS := $(BUILD)/tokenizer_probe.o $(BUILD)/tokenizer_blob.o $(TEXT_OBJECTS) $(NR_OBJECTS)
.PHONY: all
all: $(BUILD)/tokenizer_probe.bin

$(BUILD)/tokenizer_probe.o: $(MODEL)/tests/tokenizer_probe.c $(BUILD)/tokenizer_probe_fixtures.h $(MODEL)/text/tokenizer_resource.h $(MAKEFILE_LIST)
	$(RISCV_CC) $(NR_CFLAGS) -Wall -Wextra -Werror -fstack-usage -I$(MODEL)/text -I$(BUILD) -c $< -o $@

$(BUILD)/%.o: $(MODEL)/text/%.c $(MODEL)/text/tokenizer_resource.h $(MODEL)/text/unicode_tables.h $(MAKEFILE_LIST)
	$(RISCV_CC) $(NR_CFLAGS) -Wall -Wextra -Werror -fstack-usage -c $< -o $@

$(BUILD)/tokenizer_blob.o: $(BUILD)/tokenizer_blob.S $(BUILD)/tokenizer.bin $(MAKEFILE_LIST)
	$(RISCV_CC) $(NR_CFLAGS) -c $< -o $@

$(BUILD)/tokenizer_probe.elf: $(OBJECTS) $(NR_LINKER_SCRIPT) $(MAKEFILE_LIST)
	$(RISCV_LD) -T $(NR_LINKER_SCRIPT) --gc-sections -Map=$(BUILD)/tokenizer_probe.map -o $@ $(OBJECTS)

$(BUILD)/tokenizer_probe.bin: $(BUILD)/tokenizer_probe.elf
	$(RISCV_OBJCOPY) -O binary $< $@
