BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

# Directly view MLIR file output
dump:
	@${BUDDY_OPT} ./bb_mul_warp16.mlir

# Directly view MLIR file conversion result
bb_mul_warp16-lower:
	@${BUDDY_OPT} ./bb_mul_warp16.mlir \
		-lower-buckyball \
		-o log.mlir
	@cat log.mlir

bb_tile_matmul-lower:
	@${BUDDY_OPT} ./bb_tile_matmul.mlir \
		-lower-buckyball \
		-o log.mlir
	@cat log.mlir


bb_metatile_matmul-lower:
	@${BUDDY_OPT} ./bb_metatile_matmul.mlir \
		-lower-buckyball \
		-o log.mlir
	@cat log.mlir

# Clean generated files
clean:
	rm -f log.* a.out

bb_mergetile_matmul-lower:
	@${BUDDY_OPT} ./bb_mergetile_matmul.mlir \
		-lower-buckyball \
		-o log.mlir
	@cat log.mlir

bb_mvin-lower:
	@${BUDDY_OPT} ./bb_mvin.mlir \
		-lower-buckyball \
		-o log.mlir
	@cat log.mlir

bb_mvout-lower:
	@${BUDDY_OPT} ./bb_mvout.mlir \
		-lower-buckyball \
		-o log.mlir
	@cat log.mlir

.PHONY: dump bb_mul_warp16-lower bb_mul_warp16-translate bb_mul_warp16-asm bb_mul_warp16-run \
        bb_tile_matmul-lower bb_tile_matmul-translate bb_tile_matmul-asm bb_tile_matmul-run \
        bb_metatile_matmul-lower bb_metatile_matmul-translate bb_metatile_matmul-asm bb_metatile_matmul-run \
        bb_mergetile_matmul-lower bb_mergetile_matmul-translate bb_mergetile_matmul-asm bb_mergetile_matmul-run clean
