.DEFAULT_GOAL := all
CASES := $(sort $(patsubst %/kernel.mlir,%,$(wildcard */kernel.mlir)))
.PHONY: all check generate list clean $(CASES)
all: $(CASES)
$(CASES):
	$(MAKE) -C $@ all
check:
	@set -e; for c in $(CASES); do $(MAKE) -C $$c check; done
generate:
	python3 tools/generate_matmul.py
	python3 tools/generate_elementwise.py
	python3 tools/generate_quant.py
list:
	@printf '%s\n' $(CASES)
clean:
	@set -e; for c in $(CASES); do $(MAKE) -C $$c clean; done

.PHONY: suite suite-host run
suite:
	python3 tools/build_suite.py --group=$(or $(GROUP),all)
suite-host:
	python3 tools/build_suite.py --group=$(or $(GROUP),all) --host
	./build/suite-$(or $(GROUP),all)-host/suite
run: suite
	../fpga_run.sh build/suite-$(or $(GROUP),all)/suite.bin --fpga=$(or $(FPGA),5) --capture-seconds=$(or $(CAPTURE_SECONDS),1800) --completion-marker='[nr] RA returned:'
