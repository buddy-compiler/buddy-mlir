BUDDY_ROOT := ../..
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
BUDDY_LLC := $(BUDDY_ROOT)/llvm/build/bin/llc
QWEN3 ?= $(QWEN3_0_6B_DIR)
PLATFORM := $(QWEN3)/platform
TOOLS := tools

ifeq ($(QWEN3),)
  $(error QWEN3_0_6B_DIR is required: point it at the qwen3-0.6b checkout)
endif

LAYERWISE_BUILD ?= build-model-w8a8-layerwise-fixed-prompt
LAYERWISE_IMAGE_BUILD ?= build-layerwise-fpga
LAYERWISE_COMPONENTS := embedding_prefill embedding_decode \
	decoder_layer_prefill decoder_layer_decode final_head
LAYERWISE_COMPONENT_OBJS := $(addprefix $(LAYERWISE_IMAGE_BUILD)/,$(addsuffix .o,$(LAYERWISE_COMPONENTS)))
# S=128 decoder prefill currently requests 41,247,168 bytes before its
# per-layer arena reset. Keep alignment/runtime headroom without per-op heap
# growth across layers.
LAYERWISE_HEAP_SIZE ?= 50331648u
LAYERWISE_RUNTIME_TRACE ?= 0
LAYERWISE_TRACE_LIVE ?= 0
# Both values change the generated object.  Encode them in the filename so an
# incremental build cannot silently reuse a runtime compiled with a stale heap.
LAYERWISE_RUNTIME_OBJ := $(LAYERWISE_IMAGE_BUILD)/runtime-heap$(LAYERWISE_HEAP_SIZE)-trace$(LAYERWISE_RUNTIME_TRACE)-live$(LAYERWISE_TRACE_LIVE).o
LAYERWISE_CHECKPOINTS ?= 0
LAYERWISE_DECODE_STEPS ?= 1
LAYERWISE_PREFILL_LEN ?= $(shell python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["prefill_len"])' $(LAYERWISE_BUILD)/model_compile_config.json)
LAYERWISE_MAX_CACHE_LEN ?= $(shell python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_cache_len"])' $(LAYERWISE_BUILD)/model_compile_config.json)
LAYERWISE_EOS_TOKEN_ID ?= 151645
LAYERWISE_MAIN_SOURCE := layerwise_fpga_main.c
LAYERWISE_MAIN_OBJ := $(LAYERWISE_IMAGE_BUILD)/main-prefill$(LAYERWISE_PREFILL_LEN)-cache$(LAYERWISE_MAX_CACHE_LEN)-checkpoints$(LAYERWISE_CHECKPOINTS)-decode$(LAYERWISE_DECODE_STEPS).o
LAYERWISE_ELF := $(LAYERWISE_IMAGE_BUILD)/buddy_qwen3_layerwise.elf
LAYERWISE_RAW_BIN := $(LAYERWISE_IMAGE_BUILD)/buddy_qwen3_layerwise.raw.bin
LAYERWISE_TARGET := $(LAYERWISE_IMAGE_BUILD)/buddy_qwen3_layerwise.bin
LAYERWISE_RUNTIME_INPUTS := \
	$(LAYERWISE_BUILD)/prefill_input_ids_i64.data \
	$(LAYERWISE_BUILD)/prompt_length_i32.data \
	$(LAYERWISE_BUILD)/prefill_rope_cos_f32.data \
	$(LAYERWISE_BUILD)/prefill_rope_sin_f32.data \
	$(LAYERWISE_BUILD)/prefill_mask_f32.data \
	$(LAYERWISE_BUILD)/decode_rope_cos_f32.data \
	$(LAYERWISE_BUILD)/decode_rope_sin_f32.data \
	$(LAYERWISE_BUILD)/decode_mask_f32.data \
	$(LAYERWISE_BUILD)/cache_write_mask_f32.data
LAYERWISE_DATA_CFLAGS := \
	-DQWEN3_PARAMS_F32_FILE='"$(LAYERWISE_BUILD)/params_f32.data"' \
	-DQWEN3_PARAMS_I8_FILE='"$(LAYERWISE_BUILD)/params_i8.data"' \
	-DQWEN3_PREFILL_IDS_FILE='"$(LAYERWISE_BUILD)/prefill_input_ids_i64.data"' \
	-DQWEN3_PROMPT_LENGTH_FILE='"$(LAYERWISE_BUILD)/prompt_length_i32.data"' \
	-DQWEN3_PREFILL_ROPE_COS_FILE='"$(LAYERWISE_BUILD)/prefill_rope_cos_f32.data"' \
	-DQWEN3_PREFILL_ROPE_SIN_FILE='"$(LAYERWISE_BUILD)/prefill_rope_sin_f32.data"' \
	-DQWEN3_PREFILL_MASK_FILE='"$(LAYERWISE_BUILD)/prefill_mask_f32.data"' \
	-DQWEN3_DECODE_ROPE_COS_FILE='"$(LAYERWISE_BUILD)/decode_rope_cos_f32.data"' \
	-DQWEN3_DECODE_ROPE_SIN_FILE='"$(LAYERWISE_BUILD)/decode_rope_sin_f32.data"' \
	-DQWEN3_DECODE_MASK_FILE='"$(LAYERWISE_BUILD)/decode_mask_f32.data"' \
	-DQWEN3_CACHE_WRITE_MASK_FILE='"$(LAYERWISE_BUILD)/cache_write_mask_f32.data"'

RISCV_GNU_CC := $(shell command -v riscv64-linux-gnu-gcc 2>/dev/null)
ifneq ($(RISCV_GNU_CC),)
RISCV_CC := riscv64-linux-gnu-gcc
RISCV_LD := riscv64-linux-gnu-ld
RISCV_OBJCOPY := riscv64-linux-gnu-objcopy
else
RISCV_CC := clang --target=riscv64-unknown-elf
RISCV_LD := ld.lld -m elf64lriscv
RISCV_OBJCOPY := /usr/lib/llvm-19/bin/llvm-objcopy
endif
CFLAGS := -march=rv64gcv -mabi=lp64d -mcmodel=medany -nostdlib \
	-fno-builtin -ffreestanding -fno-pie -static -O2 -DFPGA_UART_4BYTE \
	-I$(PLATFORM) -I$(QWEN3)/tokenizer -I$(TOOLS)
LLCFLAGS := -filetype=obj -mtriple=riscv64 \
	-mattr=+m,+f,+d,+v,+xboscame \
	-code-model=medium
# The FPGA implements a fixed 512-bit VLEN but does not expose the optional
# vlenb CSR.  Tell LLVM the exact VLEN for the large layer-wise wrappers so
# scalable-vector stack frames are resolved to constants instead of csrr vlenb.
LAYERWISE_LLCFLAGS := $(filter-out -mattr=%,$(LLCFLAGS)) \
	-mattr=+m,+f,+d,+v,+zvl512b,+xboscame \
	-riscv-v-vector-bits-min=512 -riscv-v-vector-bits-max=512 \
	-combiner-store-merging=false
LAYERWISE_ASM_CFLAGS := $(filter-out -march=%,$(CFLAGS)) -march=rv64gcv

.PHONY: all layerwise-components layerwise-image layerwise-image-check clean

all: layerwise-image-check

layerwise-components: $(LAYERWISE_COMPONENT_OBJS)

layerwise-image: $(LAYERWISE_TARGET)

layerwise-image-check: $(LAYERWISE_ELF) $(LAYERWISE_TARGET)
	python3 verify_layerwise_image.py \
		--artifacts $(LAYERWISE_BUILD) \
		--elf $(LAYERWISE_ELF) --bin $(LAYERWISE_TARGET) \
		--heap-size $(patsubst %u,%,$(LAYERWISE_HEAP_SIZE)) \
		--decode-steps $(LAYERWISE_DECODE_STEPS)

$(LAYERWISE_IMAGE_BUILD):
	mkdir -p $@

define LAYERWISE_COMPONENT_RULES
$(LAYERWISE_IMAGE_BUILD)/$(1).s: $(LAYERWISE_BUILD)/lowering-$(1)/subgraph0_$(1).ll | $(LAYERWISE_IMAGE_BUILD)
	$$(BUDDY_LLC) $$(filter-out -filetype=obj,$$(LAYERWISE_LLCFLAGS)) \
		-filetype=asm -o $$@ $$<

$(LAYERWISE_IMAGE_BUILD)/$(1)_bare.s: $(LAYERWISE_IMAGE_BUILD)/$(1).s \
		$$(TOOLS)/rvv_fpga_compat.py $$(TOOLS)/ame_to_word.py
	sed -e '/^[[:space:]]*\.attribute[[:space:]]*5,/d' \
	    -e '/^[[:space:]]*\.cfi_/d' $$< | \
	perl -pe 's/^(\s*)vl1re8\.v\s+(v[0-9]+),/\1vsetivli\tzero, 64, e8, m1, ta, ma\n\1vle8.v\t\2,/; s/^(\s*)vl1re16\.v\s+(v[0-9]+),/\1vsetivli\tzero, 32, e16, m1, ta, ma\n\1vle16.v\t\2,/; s/^(\s*)vl1r(?:e32)?\.v\s+(v[0-9]+),/\1vsetivli\tzero, 16, e32, m1, ta, ma\n\1vle32.v\t\2,/; s/^(\s*)vl1re64\.v\s+(v[0-9]+),/\1vsetivli\tzero, 8, e64, m1, ta, ma\n\1vle64.v\t\2,/; s/^(\s*)vs1r\.v\s+(v[0-9]+),/\1vsetivli\tzero, 16, e32, m1, ta, ma\n\1vse32.v\t\2,/; s/^(\s*)vmv1r\.v\s+(v[0-9]+),\s*(v[0-9]+)/\1vsetivli\tzero, 16, e32, m1, ta, ma\n\1vmv.v.v\t\2, \3/; s/^(\s*)vsetivli\s+zero,\s*1,\s*e16,\s*m1,/\1vsetvli\tzero, zero, e16, mf2,/' | \
	perl -pe 's/^(\s*)(vse(?:8|16|32|64)\.v.*)/\1\2\n\1fence\trw, rw/' | \
	python3 $$(TOOLS)/rvv_fpga_compat.py | \
	AME_STORE_FENCE_AFTER=0 python3 $$(TOOLS)/ame_to_word.py > $$@

$(LAYERWISE_IMAGE_BUILD)/$(1).o: $(LAYERWISE_IMAGE_BUILD)/$(1)_bare.s
	$$(RISCV_CC) $$(LAYERWISE_ASM_CFLAGS) -c -o $$@ $$<
endef

$(foreach component,$(LAYERWISE_COMPONENTS),$(eval $(call LAYERWISE_COMPONENT_RULES,$(component))))

$(LAYERWISE_IMAGE_BUILD)/bindings.o: $(LAYERWISE_BUILD)/layerwise_bindings.c \
		layerwise_scheduler.h | $(LAYERWISE_IMAGE_BUILD)
	$(RISCV_CC) $(CFLAGS) -I. -c -o $@ $<

$(LAYERWISE_IMAGE_BUILD)/scheduler.o: layerwise_scheduler.c \
		layerwise_scheduler.h | $(LAYERWISE_IMAGE_BUILD)
	$(RISCV_CC) $(CFLAGS) -I. -c -o $@ $<

$(LAYERWISE_MAIN_OBJ): $(LAYERWISE_MAIN_SOURCE) \
		layerwise_scheduler.h | $(LAYERWISE_IMAGE_BUILD)
	$(RISCV_CC) $(CFLAGS) -I. \
		-DQWEN3_PREFILL_LEN=$(LAYERWISE_PREFILL_LEN) \
		-DQWEN3_MAX_CACHE_LEN=$(LAYERWISE_MAX_CACHE_LEN) \
		-DQWEN3_DECODE_STEPS=$(LAYERWISE_DECODE_STEPS) \
		-DQWEN3_EOS_TOKEN_ID=$(LAYERWISE_EOS_TOKEN_ID) \
		-DQWEN3_ENABLE_CHECKPOINTS=$(LAYERWISE_CHECKPOINTS) -c -o $@ $<

$(LAYERWISE_IMAGE_BUILD)/data.o: layerwise_fpga_data.S \
		$(LAYERWISE_BUILD)/params_f32.data \
		$(LAYERWISE_BUILD)/params_i8.data $(LAYERWISE_RUNTIME_INPUTS) \
		| $(LAYERWISE_IMAGE_BUILD)
	$(RISCV_CC) $(CFLAGS) $(LAYERWISE_DATA_CFLAGS) -c -o $@ $<

$(LAYERWISE_RUNTIME_OBJ): $(TOOLS)/bare_runtime.c | $(LAYERWISE_IMAGE_BUILD)
	$(RISCV_CC) $(CFLAGS) \
		-DBARE_RUNTIME_HEAP_SIZE=$(LAYERWISE_HEAP_SIZE) \
		-DBARE_RUNTIME_TRACE_MALLOC=$(LAYERWISE_RUNTIME_TRACE) \
		-DBARE_RUNTIME_TRACE_LIVE=$(LAYERWISE_TRACE_LIVE) -c -o $@ $<

$(LAYERWISE_IMAGE_BUILD)/math.o: layerwise_bare_math.c | $(LAYERWISE_IMAGE_BUILD)
	$(RISCV_CC) $(CFLAGS) -c -o $@ $<

$(LAYERWISE_IMAGE_BUILD)/crt.o: $(PLATFORM)/crt_uart.S | $(LAYERWISE_IMAGE_BUILD)
	$(RISCV_CC) $(CFLAGS) -c -o $@ $<

$(LAYERWISE_IMAGE_BUILD)/uart.o: $(PLATFORM)/uart.c | $(LAYERWISE_IMAGE_BUILD)
	$(RISCV_CC) $(CFLAGS) -c -o $@ $<

$(LAYERWISE_ELF): $(LAYERWISE_IMAGE_BUILD)/crt.o \
		$(LAYERWISE_MAIN_OBJ) \
		$(LAYERWISE_IMAGE_BUILD)/scheduler.o \
		$(LAYERWISE_IMAGE_BUILD)/bindings.o \
		$(LAYERWISE_COMPONENT_OBJS) \
		$(LAYERWISE_RUNTIME_OBJ) \
		$(LAYERWISE_IMAGE_BUILD)/math.o \
		$(LAYERWISE_IMAGE_BUILD)/uart.o \
		$(LAYERWISE_IMAGE_BUILD)/data.o \
		$(TOOLS)/qwen3_layerwise_compact.ld
	$(RISCV_LD) --script=$(TOOLS)/qwen3_layerwise_compact.ld \
		--gc-sections -Map=$(LAYERWISE_IMAGE_BUILD)/buddy_qwen3_layerwise.map \
		-o $@ $(filter %.o,$^)

$(LAYERWISE_RAW_BIN): $(LAYERWISE_ELF)
	$(RISCV_OBJCOPY) -O binary $< $@

$(LAYERWISE_TARGET): $(LAYERWISE_RAW_BIN)
	@N=$$($(TOOLS)/pad_to_fix.sh $< $@); \
	echo "writemem -rtl ariane_xilinx.i_ddr[$${N}:0]"; \
	echo "  -file $@ -file_type bin"

clean:
	rm -rf $(LAYERWISE_IMAGE_BUILD)
