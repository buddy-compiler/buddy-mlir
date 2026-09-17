# NH single-hart Hello World for RAV0.5 / NR.
# hello.c -> RISC-V objects -> hello.elf -> hello.bin (raw, not padded).
# The server's make uv_run5 wrapper pads the image before loading DDR.

.PHONY: all clean check dump help size
.DELETE_ON_ERROR:

COMMON          := ../common
include $(COMMON)/toolchain.mk

UART_DIR        := $(COMMON)/uart
RUNTIME_DIR     := $(COMMON)/runtime
COMMON_RUNTIME  := $(RUNTIME_DIR)/bare_runtime.c
LD_SCRIPT       := platform/hello_nr.ld
# This generic single-hart CRT clears BSS and calls the runtime's _init.
# The NR dual-core CRT dispatches to nanhu_main / _ra_init; hello needs neither.
CRT_S           := $(RUNTIME_DIR)/crt_uart.S
UART_C          := $(UART_DIR)/uart.c

SRC             := hello.c
BUILD_DIR       := build
BIN             := $(BUILD_DIR)/hello.bin
OBJECTS         := $(addprefix $(BUILD_DIR)/,crt_uart.o hello.o bare_runtime.o uart.o)

RISCV_CFLAGS    := -march=rv64gc -mabi=lp64d -mcmodel=medany -nostdlib \
                   -fno-builtin -ffreestanding -fno-pie -O2 \
                   -ffunction-sections -fdata-sections -MMD -MP \
                   -DBARE_RUNTIME_ENABLE_AME=0 \
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

$(BUILD_DIR)/hello.elf: $(OBJECTS) $(LD_SCRIPT) $(MAKEFILE_LIST)
	$(RISCV_LD) --script=$(LD_SCRIPT) --gc-sections -o $@ $(OBJECTS)

$(BIN): $(BUILD_DIR)/hello.elf $(MAKEFILE_LIST)
	$(RISCV_OBJCOPY) -O binary $< $@

size: $(BIN)
	@S=$$(stat -c%s $(BIN)); \
	echo "hello.bin: $${S} bytes (raw), NH entry: 0x80000000"; \
	if [ $$(( S % 64 )) -eq 0 ]; then \
	  echo "writemem -rtl {nanhu_xilinx.i_ddr[$$(( S / 64 - 1 )):0]} -file <hello.bin> -file_type bin"; \
	else \
	  echo "Not 64-byte aligned; use the server's make uv_run5 wrapper to pad and load."; \
	fi

# This checks application text with UART stubs, not the MMIO driver or startup.
HOST_CC ?= cc
$(BUILD_DIR)/host_check: tools/host_check.c $(SRC) $(RUNTIME_DIR)/bare_runtime.h $(UART_DIR)/uart.h $(MAKEFILE_LIST) | $(BUILD_DIR)
	$(HOST_CC) -O1 -Wall -DHELLO_NO_MAIN -isystem $(UART_DIR) -I$(RUNTIME_DIR) tools/host_check.c $(SRC) -o $@

check: $(BUILD_DIR)/host_check
	@$(BUILD_DIR)/host_check

dump: $(BUILD_DIR)/hello.elf
	$(RISCV_OBJDUMP) -h -d $<

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "  make all     Build $(BIN) for NH / NR (raw, not padded)"
	@echo "  make check   Host-side UART output check"
	@echo "  make size    Report size and loading requirements"
	@echo "  make dump    Show ELF sections and full disassembly"
	@echo "  make clean"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(OBJECTS:.o=.d)
