# Temporary NH/E6 f32-reduction workarounds. Keep SSA lowering,
# assembly filtering and final ELF auditing under one policy; opt out only
# for explicit hardware diagnosis.
FPGA_REDUCTION_POLICY_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
ALLOW_VECTOR_ORDERED_REDUCTIONS ?= 0
ifneq ($(words $(ALLOW_VECTOR_ORDERED_REDUCTIONS)),1)
$(error ALLOW_VECTOR_ORDERED_REDUCTIONS must be 0 or 1, got '$(ALLOW_VECTOR_ORDERED_REDUCTIONS)')
endif
ifeq ($(filter 0 1,$(ALLOW_VECTOR_ORDERED_REDUCTIONS)),)
$(error ALLOW_VECTOR_ORDERED_REDUCTIONS must be 0 or 1, got '$(ALLOW_VECTOR_ORDERED_REDUCTIONS)')
endif
ORDERED_REDUCTION_FLAGS := $(if $(filter 1,$(ALLOW_VECTOR_ORDERED_REDUCTIONS)),--allow-vector-ordered-reductions)
ORDERED_REDUCTION_FILTER_FLAGS := $(if $(filter 1,$(ALLOW_VECTOR_ORDERED_REDUCTIONS)),--allow-vector-ordered-reductions,--reject-vector-ordered-reductions)
FPGA_ORDERED_REDUCTION_LOWER_OPTION := scalar-f32-ordered-reductions=$(if $(filter 1,$(ALLOW_VECTOR_ORDERED_REDUCTIONS)),false,true)
ALLOW_VECTOR_MAX_REDUCTIONS ?= 0
ifneq ($(words $(ALLOW_VECTOR_MAX_REDUCTIONS)),1)
$(error ALLOW_VECTOR_MAX_REDUCTIONS must be 0 or 1, got '$(ALLOW_VECTOR_MAX_REDUCTIONS)')
endif
ifeq ($(filter 0 1,$(ALLOW_VECTOR_MAX_REDUCTIONS)),)
$(error ALLOW_VECTOR_MAX_REDUCTIONS must be 0 or 1, got '$(ALLOW_VECTOR_MAX_REDUCTIONS)')
endif
MAX_REDUCTION_FLAGS := $(if $(filter 1,$(ALLOW_VECTOR_MAX_REDUCTIONS)),--allow-vector-max-reductions)
MAX_REDUCTION_FILTER_FLAGS := $(if $(filter 1,$(ALLOW_VECTOR_MAX_REDUCTIONS)),--allow-vector-max-reductions,--reject-vector-max-reductions)
FPGA_MAX_REDUCTION_LOWER_OPTION := scalar-f32-max-reductions=$(if $(filter 1,$(ALLOW_VECTOR_MAX_REDUCTIONS)),false,true)
FPGA_REDUCTION_LOWER_OPTIONS := $(FPGA_ORDERED_REDUCTION_LOWER_OPTION) $(FPGA_MAX_REDUCTION_LOWER_OPTION)
FPGA_REDUCTION_FILTER_FLAGS := $(ORDERED_REDUCTION_FILTER_FLAGS) $(MAX_REDUCTION_FILTER_FLAGS)
FPGA_REDUCTION_AUDIT_FLAGS := $(ORDERED_REDUCTION_FLAGS) $(MAX_REDUCTION_FLAGS)
export ALLOW_VECTOR_ORDERED_REDUCTIONS ALLOW_VECTOR_MAX_REDUCTIONS

# Recheck on every invocation, but preserve the timestamp when unchanged.
# Depending on this file makes 0 -> 1 -> 0 effective without make clean, even
# for kernels whose filtered assembly is identical under both policies.
.PHONY: fpga-reduction-policy-force
fpga-reduction-policy-force:
%/.ordered-reduction-policy: fpga-reduction-policy-force $(FPGA_REDUCTION_POLICY_MAKEFILE)
	@set -eu; \
	  mkdir -p "$(@D)"; \
	  tmp="$@.tmp"; \
	  trap 'rm -f "$$tmp"' EXIT INT TERM; \
	  printf '%s\n' 'ALLOW_VECTOR_ORDERED_REDUCTIONS=$(ALLOW_VECTOR_ORDERED_REDUCTIONS)' 'ALLOW_VECTOR_MAX_REDUCTIONS=$(ALLOW_VECTOR_MAX_REDUCTIONS)' > "$$tmp"; \
	  if test -r "$@" && cmp -s "$$tmp" "$@"; then rm -f "$$tmp"; else mv -f "$$tmp" "$@"; fi; \
	  trap - EXIT INT TERM
