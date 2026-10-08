TRITON_PYTHON ?= python3
FAMILY ?= residual_add
.DEFAULT_GOAL := all
.PHONY: all check run export list generate
all:
	$(TRITON_PYTHON) build.py --family=$(FAMILY) --nr
check:
	$(TRITON_PYTHON) build.py --family=$(FAMILY) --host
run:
	$(TRITON_PYTHON) build.py --family=$(FAMILY) --run
export:
	$(TRITON_PYTHON) export.py --family=$(FAMILY)
list:
	$(TRITON_PYTHON) cases.py --family=$(FAMILY)
generate:
	$(TRITON_PYTHON) cases.py --family=$(FAMILY) --generate
