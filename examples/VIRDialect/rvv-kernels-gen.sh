#!/usr/bin/env bash

# Usage: ./rvv-kernels-gen.sh
# This script generates test cases for the RISC-V vector extension kernels.

set -u  # Avoid undefined variables, removed -e

# ==== Configuration ====
RUN=1  # 0=print only, 1=execute make

FIXED_STEPS=(4 8 16 32 64 128 256 512 1024)
SCALABLE_STEPS=(2 4 8 16 32 64 128)
VP_STEPS=(2 4 8 16 32 64 128)

SIZES=(4096 4098 131072 131074 4194304 4194306 67108864 67108866)

FIXED_TARGET="vector-saxpy-fixed-cross-rvv-aot"
SCALABLE_TARGET="vector-saxpy-scalable-cross-rvv-aot"
VP_TARGET="vector-saxpy-vp-cross-rvv-aot"

gen_cases () {
  # Extract target name and create nameref to steps array
  local target="$1"; shift
  local -n steps_ref=$1; shift

  # Initialize counter for test cases
  local count=0

  # Iterate through all data sizes
  for size in "${SIZES[@]}"; do
    # Iterate through all step sizes for current vectorization strategy
    for step in "${steps_ref[@]}"; do
      # Build make command with current parameters
      cmd="make $target STEP=$step SIZE=$size"
      echo "$cmd"

      # Execute command if RUN flag is set to 1
      if [[ "$RUN" -eq 1 ]]; then
        eval "$cmd"
      fi

      # Increment test case counter
      ((count++))
    done
  done

  # Print summary of generated test cases
  echo "===> $target generated ${count} test cases in total"
}

gen_cases "$FIXED_TARGET" FIXED_STEPS
gen_cases "$SCALABLE_TARGET" SCALABLE_STEPS
gen_cases "$VP_TARGET" VP_STEPS
