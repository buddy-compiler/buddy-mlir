.DEFAULT_GOAL := help
PYTHON ?= python3
# Python 3.11 with the repository's built Buddy bindings on PYTHONPATH.
BUDDY_PYTHON ?= /home/chh/venvs/qwen3fpga-py311/bin/python
BUDDY_PYTHONPATH ?= $(abspath $(REPO)/build-python/python_packages)
REPO := ../../../..
LLVM_BIN ?= $(REPO)/llvm/build-2d26/bin
HOST_CC ?= $(LLVM_BIN)/clang
RISCV_CC ?= $(LLVM_BIN)/clang
LLVM_AR ?= $(LLVM_BIN)/llvm-ar
LLVM_NM ?= $(LLVM_BIN)/llvm-nm
ASSETS ?= assets/official
CHECKPOINT ?= assets/checkpoint
CONTEXT ?= 128
LAYERS ?=
PROMPT ?= 151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271
IMPORT_DIR ?= build/import/full-28l-cap$(CONTEXT)
MODEL_CASES ?= build/model-cases-cap$(CONTEXT)

.PHONY: help assets checkpoint resources check inventory frontend text-library \
        archive-smoke import layout inventory-report reference host-run \
        model-cases model-archive memory-plan quant-reference \
        triton-calls model-lib weight-image replaced-host-run run-model-kernels lower-nr image-1layer tokenizer-probe

help:
	@echo "Preparation:  assets checkpoint resources check inventory frontend text-library"
	@echo "Graph import: import layout inventory-report"
	@echo "References:   reference quant-reference"
	@echo "Execution:    host-run"
	@echo "Kernels:      model-cases model-archive archive-smoke"
	@echo "Replacement:  triton-calls model-lib replaced-host-run"
	@echo "Planning:     memory-plan"
	@echo "Image:        weight-image"
	@echo "Text probe:   tokenizer-probe (local build/audit only; no board launch)"
	@echo "Board:        run-model-kernels (needs the FPGA server)"
	@echo ""
	@echo "The import/reference/host-run targets need BUDDY_PYTHON (Python 3.11 with"
	@echo "the repository's Buddy bindings). They never touch the FPGA."

# ---------------------------------------------------------------- resources
assets:
	$(PYTHON) -B tools/fetch_assets.py --output $(ASSETS)

# Revision-pinned weights with resumable transfer and post-download SHA256.
checkpoint:
	$(PYTHON) -B tools/fetch_checkpoint.py --output $(CHECKPOINT) \
	  --manifest validation/checkpoint-manifest.json

resources:
	$(PYTHON) -B tools/pack_tokenizer.py --assets $(ASSETS) --output build/tokenizer.bin

check:
	$(PYTHON) -B tools/check_text.py --assets $(ASSETS) --build build --output validation/text-check.json --cc '$(HOST_CC)'
	$(PYTHON) -B tools/check_tokenizer_encode.py --assets $(ASSETS) --build build/text-encode-audit --output validation/tokenizer-encode-audit.json --cc '$(HOST_CC)'
	$(PYTHON) -B -m unittest discover -s tests -p 'test_*.py' -v

inventory: resources
	$(PYTHON) -B tools/checkpoint_inventory.py --assets $(ASSETS) --tokenizer-blob build/tokenizer.bin --context $(CONTEXT) --output validation

frontend:
	$(PYTHON) -B tools/inspect_frontend.py --assets $(ASSETS) --meta-check --output validation/frontend-environment.json

text-library:
	mkdir -p build
	$(RISCV_CC) --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -mcmodel=medany -O2 -ffreestanding -fno-builtin -fno-vectorize -fno-slp-vectorize -Wall -Wextra -Werror -c text/tokenizer_resource.c -o build/tokenizer_resource.o
	$(RISCV_CC) --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -mcmodel=medany -O2 -ffreestanding -fno-builtin -fno-vectorize -fno-slp-vectorize -Wall -Wextra -Werror -c text/tokenizer_encode.c -o build/tokenizer_encode.o
	$(RISCV_CC) --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -mcmodel=medany -O2 -ffreestanding -fno-builtin -fno-vectorize -fno-slp-vectorize -Wall -Wextra -Werror -c text/unicode_tables.c -o build/unicode_tables.o
	$(LLVM_AR) rcsD build/libqwen_text.a build/tokenizer_resource.o build/tokenizer_encode.o build/unicode_tables.o

tokenizer-probe:
	$(PYTHON) -B tools/build_tokenizer_probe.py --assets $(ASSETS) --llvm-bin $(LLVM_BIN)

# ------------------------------------------------------------- graph import
# Real Qwen3 -> Buddy graph. Saves the op dump, the typed IR and the params, and
# warms up the StaticCache so prefill and decode share one cache-as-input ABI.
import:
	PYTHONPATH=$(BUDDY_PYTHONPATH) $(BUDDY_PYTHON) -B tools/import_model.py \
	  --assets $(ASSETS) --checkpoint $(CHECKPOINT) --output $(IMPORT_DIR) \
	  --prefill-len 16 --max-cache-len $(CONTEXT) --fuse none --save-mlir --init-cache \
	  $(if $(LAYERS),--layers $(LAYERS),)

# Deployment weight offsets, derived from the compiled main graph and matched to
# checkpoint tensors by content.
layout:
	$(PYTHON) -B tools/weight_layout.py \
	  --mlir $(IMPORT_DIR)/forward_decode.mlir \
	  --params $(IMPORT_DIR)/params.json \
	  --checkpoint $(CHECKPOINT)/model.safetensors \
	  --config $(ASSETS)/config.json \
	  --output validation/weight-layout.json

inventory-report:
	$(PYTHON) -B tools/operator_inventory.py \
	  --prefill $(IMPORT_DIR)/prefill-raw-ops.json \
	  --decode $(IMPORT_DIR)/decode-raw-ops.json \
	  --output validation/operator-inventory.json --layers 28

# W8A8 weight image plus its DDR placement manifest. Derived from the verified
# layout and the verified quantize contract; not uploaded anywhere.
weight-image: resources
	$(PYTHON) -B tools/build_weight_image.py --layout validation/weight-layout.json \
	  --checkpoint $(CHECKPOINT)/model.safetensors --config $(ASSETS)/config.json \
	  --output build/weight-image --cache-len $(CONTEXT) \
	  --tokenizer-blob build/tokenizer.bin

memory-plan:
	$(PYTHON) -B tools/memory_plan.py --layout validation/weight-layout.json \
	  --output validation/memory-plan.json --cache-len $(CONTEXT)

# --------------------------------------------------------------- references
# Authoritative FP32 reference (transformers on the official checkpoint).
reference:
	$(BUDDY_PYTHON) -B tools/host_reference.py \
	  --assets $(ASSETS) --checkpoint $(CHECKPOINT) \
	  --output build/reference/fp32-16plus8 --prompt-ids '$(PROMPT)' \
	  --decode-steps 8 --max-cache-len $(CONTEXT) $(if $(LAYERS),--layers $(LAYERS),)

# W8A8 reference. Add NO_QUANTIZE=1 for the exact-weight control run that
# isolates implementation error from quantization error.
quant-reference:
	$(BUDDY_PYTHON) -B tools/quant_model_reference.py \
	  --assets $(ASSETS) --checkpoint $(CHECKPOINT) --layout validation/weight-layout.json \
	  --output build/quant-reference/$(if $(NO_QUANTIZE),control,full)-28l \
	  --fp32-reference build/reference/fp32-16plus8 --prompt-ids '$(PROMPT)' \
	  --decode-steps 8 --max-cache-len $(CONTEXT) $(if $(NO_QUANTIZE),--no-quantize,) \
	  $(if $(LAYERS),--layers $(LAYERS),)

# ---------------------------------------------------------------- execution
# Runs the compiled graph on the host through MLIR's execution engine and scores
# it against the FP32 reference. This is the stage C/D check without a board.
host-run:
	PYTHONPATH=$(BUDDY_PYTHONPATH) $(BUDDY_PYTHON) -B tools/run_graph_host.py \
	  --assets $(ASSETS) --checkpoint $(CHECKPOINT) --layout validation/weight-layout.json \
	  --output build/host-run/full-28l --reference-dir build/reference/fp32-16plus8 \
	  --prompt-ids '$(PROMPT)' --decode-steps 8 --max-cache-len $(CONTEXT) \
	  $(if $(LAYERS),--layers $(LAYERS),)

# ------------------------------------------------- graph -> external calls
# Structural match on the real graph; shapes with no built kernel case are
# reported as uncovered rather than replaced.
triton-calls:
	PYTHONPATH=$(BUDDY_PYTHONPATH) $(BUDDY_PYTHON) -B tools/triton_call_replace.py \
	  --assets $(ASSETS) --checkpoint $(CHECKPOINT) \
	  --triton-build ../triton/build --output build/triton-call/full-28l \
	  --max-cache-len $(CONTEXT) $(if $(LAYERS),--layers $(LAYERS),)

# Static library the graph calls into (28 kernels) plus the generated adapters,
# linked against a probe that references every replaced symbol.
model-lib:
	ADAPTERS=$(abspath build/triton-call/full-28l/qwen_triton_adapters.c) \
	  LLVM_BIN=$(abspath $(LLVM_BIN)) tools/build_model_lib.sh

# Execute the graph whose operators have been replaced by external Triton calls.
triton-calls-host: triton-calls
	ADAPTERS=$(abspath build/triton-call/full-28l/qwen_triton_adapters.c) tools/build_host_bridge.sh

replaced-host-run: triton-calls-host
	PYTHONPATH=$(BUDDY_PYTHONPATH) $(BUDDY_PYTHON) -B tools/run_graph_host.py \
	  --assets $(ASSETS) --checkpoint $(CHECKPOINT) --layout validation/weight-layout.json \
	  --output build/host-run/replaced-28l --reference-dir build/reference/fp32-16plus8 \
	  --prompt-ids '$(PROMPT)' --decode-steps 8 --max-cache-len $(CONTEXT) \
	  --opt-level 1 --threads 8 --replace --triton-build ../triton/build \
	  --external-lib $(abspath build/host-bridge/libqwen_triton_host.so)

# Lower the model graph toward an NR binary and report what still blocks linking.
lower-nr:
	PYTHONPATH=$(BUDDY_PYTHONPATH) $(BUDDY_PYTHON) -B tools/lower_model_nr.py \
	  --assets $(ASSETS) --checkpoint $(CHECKPOINT) --triton-build ../triton/build \
	  --output build/nr-lower/layer$(or $(LAYERS),1) --kind $(or $(KIND),prefill) \
	  --max-cache-len $(CONTEXT) \
	  $(if $(LAYERS),--layers $(LAYERS),)

# Bare-metal image for the compiled model graph (1 layer), plus the parameter
# segment and the DDR load plan. Builds locally; does not upload.
image-1layer: triton-calls
	$(PYTHON) -B tools/lower_model_nr.py --assets $(ASSETS) --checkpoint $(CHECKPOINT) \
	  --triton-build ../triton/build --output build/board/1layer-prefill \
	  --layers 1 --kind prefill --max-cache-len $(CONTEXT)
	$(PYTHON) -B tools/build_model_image.py --repo-root $(abspath $(REPO)) \
	  --layout build/import/probe-1layer/weight-layout.json \
	  --graph-ir build/board/1layer-prefill/forward_prefill.ll \
	  --archive build/model-lib/libqwen_triton.a \
	  --adapters build/triton-call/full-28l/qwen_triton_adapters.c \
	  --output build/board/image-1layer --layers 1 --max-cache-len $(CONTEXT)

# Run the model-shape kernels on the board. Sequential; one UART owner at a time.
run-model-kernels:
	FPGA=$(FPGA) CAPTURE=$(CAPTURE) tools/run_model_kernels_fpga.sh

# ------------------------------------------------------------ kernel library
# Model-shape Triton cases, generated outside the shared operator tree and found
# by the shared build through QWEN_CASE_ROOTS.
model-cases:
	$(PYTHON) -B tools/model_kernel_cases.py --output $(MODEL_CASES) --capacity $(CONTEXT)

# Snapshot the NR-built model kernels into one static library with symbol,
# object and link evidence. Requires the model cases to be built first:
#   QWEN_CASE_ROOTS=$(abspath $(MODEL_CASES)) $(MAKE) -C ../triton all ...
model-archive:
	$(PYTHON) -B tools/archive_kernels.py --triton-build ../triton/build \
	  $(foreach c,$(MODEL_ARCHIVE_CASES),--case $(c)) \
	  --output build/model-archive --ar '$(LLVM_AR)' --nm '$(LLVM_NM)'
	$(RISCV_CC) --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -mcmodel=medany \
	  -ffreestanding -O2 -Ibuild/model-archive -c tests/model_archive_link_probe.c \
	  -o build/model-archive/model-link-probe.o
	ld.lld -m elf64lriscv -r -Map=build/model-archive/model-link-probe.map \
	  -o build/model-archive/model-linked-probe.o \
	  build/model-archive/model-link-probe.o build/model-archive/libqwen_triton.a
	$(LLVM_NM) --extern-only --defined-only build/model-archive/model-linked-probe.o
	$(LLVM_NM) --undefined-only build/model-archive/model-linked-probe.o

archive-smoke:
	$(PYTHON) -B tools/archive_kernels.py --triton-build ../triton/build --case matmul_3x19x70 --case add_1x1024 --output build/archive-$(or $(SNAPSHOT),smoke) --ar '$(LLVM_AR)' --nm '$(LLVM_NM)'
	$(RISCV_CC) --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -mcmodel=medany -ffreestanding -O2 -Ibuild/archive-$(or $(SNAPSHOT),smoke) -c tests/archive_link_probe.c -o build/archive-$(or $(SNAPSHOT),smoke)/link-probe.o
	ld.lld -m elf64lriscv -r -Map=build/archive-$(or $(SNAPSHOT),smoke)/link-probe.map -o build/archive-$(or $(SNAPSHOT),smoke)/linked-probe.o build/archive-$(or $(SNAPSHOT),smoke)/link-probe.o build/archive-$(or $(SNAPSHOT),smoke)/libqwen_triton.a
	$(LLVM_NM) --extern-only build/archive-$(or $(SNAPSHOT),smoke)/linked-probe.o
