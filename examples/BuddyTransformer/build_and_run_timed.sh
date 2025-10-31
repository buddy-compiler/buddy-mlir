#!/bin/bash
# ===- build_and_run_timed.sh ----------------------------------------------
#
# Script to build and run the timed transformer executable
#
# ===---------------------------------------------------------------------------

set -e  # Exit on error

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUDDY_ROOT="$SCRIPT_DIR/../.."
BUILD_DIR="$BUDDY_ROOT/build"

echo "========================================="
echo "  Building Timed Transformer Executable"
echo "========================================="
echo ""
echo "Buddy-MLIR Root: $BUDDY_ROOT"
echo "Build Directory: $BUILD_DIR"
echo ""

# Step 1: Generate MLIR files and parameters
echo "Generating MLIR files and parameters..."
cd "$SCRIPT_DIR"
python import-transformer.py --output-dir ./
echo "✓ Generated forward.mlir, subgraph0.mlir, arg0.data"
echo ""

# Step 2: Reconfigure CMake to pick up new targets
echo "Reconfiguring CMake..."
cd "$BUILD_DIR"
cmake .. -G Ninja \
  -DBUDDY_TRANSFORMER_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
echo "✓ CMake reconfigured"
echo ""

# Step 3: Build the project
echo "Building the project..."
ninja buddy-transformer-timed-executable
echo "✓ Build completed"
echo ""

# Step 4: Copy parameters to build directory
echo "Copying parameters to build directory..."
cp "$SCRIPT_DIR/arg0.data" "$BUILD_DIR/examples/BuddyTransformer/"
echo "✓ Parameters copied"
echo ""

# Step 5: Run the timed executable
echo "Running timed transformer executable..."
echo "========================================="
echo ""
"$BUILD_DIR/bin/transformer-runner-timed"



