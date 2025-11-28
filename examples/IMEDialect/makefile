#!/bin/bash
# ==============================================================================
# IMEDialect Examples - Makefile
# ==============================================================================
# 
# This makefile demonstrates how to use IME dialect operations.
# Since LLVM backend does not support IME instructions yet, we only verify
# that the dialect can parse and print IME operations correctly.
#
# Usage:
#   make vmadot-basic     # Parse and verify vmadot example
#   make vfmadot-basic    # Parse and verify vfmadot example
#   make vmadot-variants  # Parse and verify vmadot variants
#   make all              # Verify all examples
#   make clean            # Remove generated files
# ==============================================================================

BUDDY_OPT := ../../build/bin/buddy-opt

# ==============================================================================
# Targets
# ==============================================================================

.PHONY: all clean vmadot-basic vfmadot-basic vmadot-variants help

all: vmadot-basic vfmadot-basic vmadot-variants
	@echo ""
	@echo "=== All IME dialect examples verified successfully! ==="

# Basic vmadot example (int8 matrix multiply)
vmadot-basic: vmadot-basic.mlir
	@echo "=== Verifying vmadot-basic.mlir ==="
	@echo "Parsing and printing IME operations..."
	$(BUDDY_OPT) $< -o vmadot-basic-out.mlir
	@echo "✓ vmadot-basic.mlir parsed successfully!"
	@echo ""

# Basic vfmadot example (fp16 matrix multiply)
vfmadot-basic: vfmadot-basic.mlir
	@echo "=== Verifying vfmadot-basic.mlir ==="
	@echo "Parsing and printing IME operations..."
	$(BUDDY_OPT) $< -o vfmadot-basic-out.mlir
	@echo "✓ vfmadot-basic.mlir parsed successfully!"
	@echo ""

# vmadot variants example (signed/unsigned combinations)
vmadot-variants: vmadot-variants.mlir
	@echo "=== Verifying vmadot-variants.mlir ==="
	@echo "Parsing and printing IME operations..."
	$(BUDDY_OPT) $< -o vmadot-variants-out.mlir
	@echo "✓ vmadot-variants.mlir parsed successfully!"
	@echo ""

# Show the parsed output
show-%: %.mlir
	@echo "=== Parsed output for $< ==="
	$(BUDDY_OPT) $<

clean:
	rm -f *-out.mlir

# Help target
help:
	@echo "IMEDialect Examples Makefile"
	@echo ""
	@echo "This makefile verifies that IME dialect operations can be"
	@echo "correctly parsed and printed by buddy-opt."
	@echo ""
	@echo "Note: LLVM backend does not yet support IME instructions,"
	@echo "      so we cannot generate actual RISC-V assembly."
	@echo ""
	@echo "Targets:"
	@echo "  vmadot-basic     - Verify basic int8 matrix multiply example"
	@echo "  vfmadot-basic    - Verify basic fp16 matrix multiply example"
	@echo "  vmadot-variants  - Verify signed/unsigned variants example"
	@echo "  all              - Verify all examples"
	@echo "  show-<name>      - Show parsed output for an example"
	@echo "  clean            - Remove generated files"
	@echo "  help             - Show this help message"
