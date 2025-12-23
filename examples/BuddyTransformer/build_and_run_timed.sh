#!/bin/bash
# ===- build_and_run_timed.sh ----------------------------------------------
#
# Script to build and run the timed transformer executable
#
# Usage: ./build_and_run_timed.sh [prefill|decode]
#
# ===---------------------------------------------------------------------------

set -e  # Exit on error

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUDDY_ROOT="$SCRIPT_DIR/../.."
BUILD_DIR="$BUDDY_ROOT/build"

# Parse stage argument (default: prefill)
STAGE="${1:-prefill}"

if [[ "$STAGE" != "prefill" && "$STAGE" != "decode" ]]; then
    echo "Usage: $0 [prefill|decode]"
    echo "  prefill: Test prefill stage (seq_len=40)"
    echo "  decode:  Test decode stage (seq_len=1)"
    exit 1
fi

echo "========================================="
echo "  Building Timed Transformer Executable"
echo "  Stage: $STAGE"
echo "========================================="
echo ""
echo "Buddy-MLIR Root: $BUDDY_ROOT"
echo "Build Directory: $BUILD_DIR"
echo ""

# Step 1: Reconfigure CMake to pick up targets
echo "Reconfiguring CMake..."
cd "$BUILD_DIR"
cmake .. -G Ninja \
  -DBUDDY_TRANSFORMER_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
echo "✓ CMake reconfigured"
echo ""

# Step 2: Build the appropriate target based on stage
echo "Building the project for $STAGE stage..."
if [[ "$STAGE" == "decode" ]]; then
    ninja buddy-transformer-decode-timed-executable
    EXECUTABLE="$BUILD_DIR/bin/transformer-runner-decode-timed"
    SEQ_LEN=1
else
    ninja buddy-transformer-prefill-timed-executable
    EXECUTABLE="$BUILD_DIR/bin/transformer-runner-prefill-timed"
    SEQ_LEN=40
fi
echo "✓ Build completed"
echo ""

# Step 3: Copy parameters to build directory (if needed)
if [[ -f "$SCRIPT_DIR/arg0.data" ]]; then
    echo "Copying parameters to build directory..."
    cp "$SCRIPT_DIR/arg0.data" "$BUILD_DIR/examples/BuddyTransformer/"
    echo "✓ Parameters copied"
    echo ""
fi

# Step 4: Run the timed executable with appropriate seq_len
echo "Running timed transformer executable ($STAGE stage, seq_len=$SEQ_LEN)..."
echo "========================================="
echo ""
TRANSFORMER_SEQ_LEN=$SEQ_LEN "$EXECUTABLE"



