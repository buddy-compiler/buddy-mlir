# Include common/toolchain.mk first (or provide RISCV_CC explicitly).
NR_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
NR_BUILD_DIR ?= build/nr
NR_LINKER_SCRIPT := $(NR_DIR)/nr.ld
NR_CFLAGS := --target=riscv64-unknown-elf -march=rv64gc_zicbom -mabi=lp64d \
  -mcmodel=medany -ffreestanding -fno-builtin -fno-pie -fno-stack-protector \
  -fno-vectorize -fno-slp-vectorize -ffp-contract=off \
  -ffunction-sections -fdata-sections -O2 -I$(NR_DIR) -I$(NR_DIR)/../uart
NR_OBJECTS := $(addprefix $(NR_BUILD_DIR)/,crt.o nr_runtime.o nr_math.o ame_sync.o nr_copy.o)

$(NR_BUILD_DIR):
	mkdir -p $@

$(NR_BUILD_DIR)/crt.o: $(NR_DIR)/crt.S | $(NR_BUILD_DIR)
	$(RISCV_CC) $(NR_CFLAGS) -c $< -o $@

$(NR_BUILD_DIR)/nr_copy.o: $(NR_DIR)/nr_copy.S | $(NR_BUILD_DIR)
	$(RISCV_CC) $(NR_CFLAGS) -march=rv64gcv -c $< -o $@

$(NR_BUILD_DIR)/%.o: $(NR_DIR)/%.c $(NR_DIR)/nr_runtime.h $(NR_DIR)/../uart/uart.h | $(NR_BUILD_DIR)
	$(RISCV_CC) $(NR_CFLAGS) -c $< -o $@
