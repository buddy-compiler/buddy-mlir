# ===- makefile ---------------------------------------------------------------
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
# NH single-hart Hello World for RAV0.5 / NR.
# hello.c + common UART/CRT/runtime -> RISC-V objects -> hello.elf ->
# hello.bin (raw, not padded). The server's make uv_runN pads to 64-byte
# DDR beats before loading. Tools come from ../common/toolchain.mk.
#
# ===---------------------------------------------------------------------------

.PHONY: all clean dump help size
.DELETE_ON_ERROR:

# Shared platform: toolchain.mk, uart/, runtime/ (crt_uart.S, bare_runtime).
COMMON          := ../common
include $(COMMON)/toolchain.mk

UART_DIR        := $(COMMON)/uart
RUNTIME_DIR     := $(COMMON)/runtime
COMMON_RUNTIME  := $(RUNTIME_DIR)/bare_runtime.c
LD_SCRIPT       := platform/hello_nr.ld
# Single-hart CRT: clear BSS, jump to C _init. Not the dual-core NR CRT
# (nanhu_main / _ra_init); hello does not start RA.
CRT_S           := $(RUNTIME_DIR)/crt_uart.S
UART_C          := $(UART_DIR)/uart.c

SRC             := hello.c
BUILD_DIR       := build
BIN             := $(BUILD_DIR)/hello.bin
OBJECTS         := $(addprefix $(BUILD_DIR)/,crt_uart.o hello.o bare_runtime.o uart.o)

# rv64gc + lp64d: matches crt_uart.S RV64 / FPU bring-up.
# -nostdlib -ffreestanding: no libc. medany for 0x80000000.
# UART_* are passed into bare_runtime.c; uart.c currently ignores them.
RISCV_CFLAGS    := -march=rv64gc -mabi=lp64d -mcmodel=medany -nostdlib \
                   -fno-builtin -ffreestanding -fno-pie -O2 \
                   -ffunction-sections -fdata-sections -MMD -MP \
                   -DUART_FREQ_HZ=14745600u -DUART_BAUD=115200u \
                   -I$(UART_DIR) -I$(RUNTIME_DIR)

all: $(BIN)

$(BUILD_DIR)/hello.o: $(SRC) $(MAKEFILE_LIST) | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/bare_runtime.o: $(COMMON_RUNTIME) $(MAKEFILE_LIST) | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/crt_uart.o: $(CRT_S) $(MAKEFILE_LIST) | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

$(BUILD_DIR)/uart.o: $(UART_C) $(MAKEFILE_LIST) | $(BUILD_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -c $< -o $@

# Link with hello_nr.ld. --gc-sections drops unused input sections.
$(BUILD_DIR)/hello.elf: $(OBJECTS) $(LD_SCRIPT) $(MAKEFILE_LIST)
	$(RISCV_LD) --script=$(LD_SCRIPT) --gc-sections -o $@ $(OBJECTS)

# Raw binary for fpga_run.sh / UVHS; ELF headers stripped.
$(BIN): $(BUILD_DIR)/hello.elf $(MAKEFILE_LIST)
	$(RISCV_OBJCOPY) -O binary $< $@

# Raw binary for fpga_run.sh. The remote UVHS uv_runN target pads to
# 64-byte DDR beats before loading.
size: $(BIN)
	@S=$$(stat -c%s $(BIN)); \
	echo "hello.bin: $${S} bytes (raw), NH entry: 0x80000000"; \
	echo "Upload with ../fpga_run.sh; remote uv_runN pads the image for DDR."

dump: $(BUILD_DIR)/hello.elf
	$(RISCV_OBJDUMP) -h -d $<

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "  make all     Build $(BIN) for NH / NR (raw, not padded)"
	@echo "  make size    Report size and loading requirements"
	@echo "  make dump    Show ELF sections and full disassembly"
	@echo "  make clean"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(OBJECTS:.o=.d)
