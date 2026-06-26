BUDDY_OPT ?= ../../build/bin/buddy-opt
BUDDY_TRANSLATE ?= ../../build/bin/buddy-translate
BUDDY_LLC ?= ../../build/bin/buddy-llc
LINUX_RISCV_GCC ?= riscv64-unknown-linux-gnu-gcc
OUT_DIR ?= /tmp/buckyball-dialect-examples

STEMS := \
  bb_relu \
  bb_quant \
  bb_dequant \
  bb_systolic \
  bb_mul_warp16 \
  bb_transpose \
  bb_im2col \
  bb_tile_matmul

.PHONY: all clean $(STEMS)

all: $(addprefix $(OUT_DIR)/,$(addsuffix -linux,$(STEMS)))

$(STEMS): %: $(OUT_DIR)/%-linux

$(OUT_DIR):
	mkdir -p $@

$(OUT_DIR)/%-linux: %.mlir debug_helper.c | $(OUT_DIR)
	$(BUDDY_OPT) $< \
	  -convert-tile-to-buckyball \
	  -lower-buckyball-to-bank-ssa \
	  -assign-physical-banks \
	  -lower-bank-ssa-to-intrinsics \
	  -lower-buckyball -expand-strided-metadata -memref-expand \
	  -convert-scf-to-cf -convert-to-llvm -convert-cf-to-llvm \
	  -reconcile-unrealized-casts \
	  -o $(OUT_DIR)/$*.llvm.mlir
	$(BUDDY_TRANSLATE) $(OUT_DIR)/$*.llvm.mlir --buddy-to-llvmir \
	  -o $(OUT_DIR)/$*.ll
	$(BUDDY_LLC) $(OUT_DIR)/$*.ll -filetype=obj -mtriple=riscv64 \
	  -mattr=+buddyext,+D -float-abi=hard -o $(OUT_DIR)/$*.o
	$(LINUX_RISCV_GCC) -c debug_helper.c -o $(OUT_DIR)/$*_debug_helper.o -O2
	$(LINUX_RISCV_GCC) $(OUT_DIR)/$*.o $(OUT_DIR)/$*_debug_helper.o \
	  -O2 -static -o $@

clean:
	rm -rf $(OUT_DIR)
