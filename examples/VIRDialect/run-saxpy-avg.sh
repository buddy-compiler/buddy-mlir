#!/usr/bin/env bash

# Usage: ./run-saxpy-avg.sh
# This script runs the saxpy kernel and calculates the average time.
# It writes the results to a CSV file.

set -u  # Exit on undefined variables

# Configuration parameters
REPEAT=5                       # Number of times to run each kernel
LOG_FILE="saxpy_time_avg.csv"  # Output CSV file for results
PATTERN="saxpy.*.*.*.out"      # Pattern to match executable files
WARMUP=1                       # Enable warmup run (1=yes, 0=no)

# Initialize CSV file with header
echo "kernel,type,step,size,time_avg_s,runs" > "$LOG_FILE"

# Enable nullglob to handle cases where no files match the pattern
shopt -s nullglob

# Process each matching executable file
for exe in $PATTERN; do
  base=$(basename "$exe")
  # Parse filename: saxpy.{type}.{step}.{size}.out
  IFS='.' read -r prefix type step size ext <<< "$base"

  # Validate file format - must start with "saxpy" and end with ".out"
  if [[ "$prefix" != "saxpy" || "$ext" != "out" ]]; then
    echo "Skipping file: $base"
    continue
  fi

  # Ensure executable has proper permissions
  if [[ ! -x "$exe" ]]; then
    chmod +x "$exe" 2>/dev/null || true
  fi

  echo "==> Running $base  (type=$type, step=$step, size=$size)"

  # Perform warmup run to avoid cold start effects
  if [[ $WARMUP -eq 1 ]]; then
    ./"$exe" >/dev/null 2>&1 || true
  fi

  # Run the kernel multiple times and collect timing data
  times=()
  for ((i=1;i<=REPEAT;i++)); do
    t=$("./$exe" 2>&1 || true)
    echo "  run $i: $t"
    times+=("$t")
  done

  # Calculate average execution time from collected data
  sum=0
  count=0
  for t in "${times[@]}"; do
    if [[ -n "$t" ]]; then
      # Use awk for high-precision floating point arithmetic
      sum=$(awk -v a="$sum" -v b="$t" 'BEGIN{printf "%.9f", a+b}')
      count=$((count+1))
    fi
  done
  if [[ $count -eq 0 ]]; then
    avg="NA"
  else
    avg=$(awk -v s="$sum" -v c="$count" 'BEGIN{printf "%.6f", s/c}')
  fi

  # Output results and write to CSV file
  echo "  ==> avg: $avg s"
  echo "${base},${type},${step},${size},${avg},${REPEAT}" >> "$LOG_FILE"
done

echo
echo "Completed. Results written to: $LOG_FILE"
