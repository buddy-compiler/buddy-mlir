.DEFAULT_GOAL := all
REPO_ROOT := ../../../../..
include ../../toolchain.mk
NR_BUILD_DIR := build/runtime
include ../nr.mk

.PHONY: all clean
all: build/rvv_probe.bin build/copy_probe.bin
build:
	mkdir -p $@
build/launch.o: launch.c ../nr_runtime.h | build
	$(RISCV_CC) $(NR_CFLAGS) -c $< -o $@
build/rvv_ops.o: rvv_ops.S | build
	$(RISCV_CC) --target=riscv64-unknown-elf -march=rv64gcv -mabi=lp64d -mcmodel=medany -c $< -o $@
build/copy_embedded.o: copy_launch.c ../nr_runtime.h | build
	$(RISCV_CC) $(NR_CFLAGS) -DNR_COPY_PROBE_EMBEDDED -c $< -o $@
build/rvv_probe.elf: build/launch.o build/rvv_ops.o build/copy_embedded.o $(NR_OBJECTS) $(NR_LINKER_SCRIPT)
	$(RISCV_LD) -T $(NR_LINKER_SCRIPT) --gc-sections -o $@ $(filter %.o,$^)
build/rvv_probe.bin: build/rvv_probe.elf
	$(RISCV_OBJCOPY) -O binary $< $@
build/copy_launch.o: copy_launch.c ../nr_runtime.h | build
	$(RISCV_CC) $(NR_CFLAGS) -c $< -o $@
build/copy_probe.elf: build/copy_launch.o $(NR_OBJECTS) $(NR_LINKER_SCRIPT)
	$(RISCV_LD) -T $(NR_LINKER_SCRIPT) --gc-sections -o $@ $(filter %.o,$^)
build/copy_probe.bin: build/copy_probe.elf
	$(RISCV_OBJCOPY) -O binary $< $@
build/uart_rx_launch.o: uart_rx_launch.c ../nr_runtime.h | build
	$(RISCV_CC) $(NR_CFLAGS) -c $< -o $@
build/uart_rx_probe.elf: build/uart_rx_launch.o $(NR_OBJECTS) $(NR_LINKER_SCRIPT)
	$(RISCV_LD) -T $(NR_LINKER_SCRIPT) --gc-sections -Map=$@.map -o $@ $(filter %.o,$^)
build/uart_rx_probe.bin: build/uart_rx_probe.elf
	$(RISCV_OBJCOPY) -O binary $< $@
build/console_wrap_launch.o: console_wrap_launch.c ../nr_runtime.h | build
	$(RISCV_CC) $(NR_CFLAGS) -c $< -o $@
build/console_wrap_probe.elf: build/console_wrap_launch.o $(NR_OBJECTS) $(NR_LINKER_SCRIPT)
	$(RISCV_LD) -T $(NR_LINKER_SCRIPT) --gc-sections -Map=$@.map -o $@ $(filter %.o,$^)
build/console_wrap_probe.bin: build/console_wrap_probe.elf
	$(RISCV_OBJCOPY) -O binary $< $@
build/ddr_launch.o: ddr_launch.c ../nr_runtime.h | build
	$(RISCV_CC) $(NR_CFLAGS) -c $< -o $@
build/ddr_probe.elf: build/ddr_launch.o $(NR_OBJECTS) $(NR_LINKER_SCRIPT)
	$(RISCV_LD) -T $(NR_LINKER_SCRIPT) --gc-sections -Map=$@.map -o $@ $(filter %.o,$^)
build/ddr_probe.bin: build/ddr_probe.elf
	$(RISCV_OBJCOPY) -O binary $< $@
clean:
	rm -rf build
