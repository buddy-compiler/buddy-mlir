#!/usr/bin/env bash

# Usage: ./fixed-kernels-gen.sh
# This script generates test cases for the AVX2 / AVX512 / ARM NEON kernels.

set -u  # Avoid undefined variables, removed -e

# ==== Configuration ====
RUN=1  # 0=print only, 1=execute make

FIXED_STEPS=(4 8 16 32 64 128 256 512 1024)

SIZES=(4096 4098 131072 131074 4194304 4194306 67108864 67108866)

FIXED_TARGET="vector-saxpy-fixed-aot"

gen_cases () {
  # Extract target name and steps array name
  local target="$1"; shift
  local steps_array_name="$1"; shift

  # Initialize counter for test cases
  local count=0

  # Iterate through all data sizes
  for size in "${SIZES[@]}"; do
    # Iterate through all step sizes for current vectorization strategy
    # Use indirect expansion to access the array
    for step in $(eval "echo \${$steps_array_name[@]}"); do
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
