.DEFAULT_GOAL := all
TRITON_PYTHON ?= python3
GROUP ?= all
JOBS ?= 4
FPGA ?= 5
CAPTURE_SECONDS ?= 1800
SELECT := $(if $(CASE),--case=$(CASE),--group=$(GROUP))
SUITE := $(if $(CASE),,--suite)
IMAGE := $(if $(CASE),build/$(CASE)/nr/$(CASE).bin,build/suite-$(GROUP)/suite.bin)

.PHONY: all export check list run
all:
	$(TRITON_PYTHON) build.py $(SELECT) $(SUITE) --jobs=$(JOBS)
export:
	$(TRITON_PYTHON) export.py $(if $(CASE),--case=$(CASE),)
check:
	$(TRITON_PYTHON) build.py $(SELECT) $(SUITE) --host --jobs=$(JOBS)
list:
	$(TRITON_PYTHON) export.py --list
run: all
	../../fpga_run.sh $(IMAGE) --fpga=$(FPGA) --capture-seconds=$(CAPTURE_SECONDS) --completion-marker='[nr] RA returned:'
