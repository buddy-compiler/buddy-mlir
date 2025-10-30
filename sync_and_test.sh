#!/bin/bash

# One-click sync Python files to build directory
# Usage: ./sync_and_test.sh

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color


print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


if [ ! -f "CMakeLists.txt" ] || [ ! -d "frontend/Python" ]; then
    print_error "Please run this script in the buddy-mlir root directory"
    exit 1
fi

# Set paths
BUDDY_ROOT=$(pwd)
BUILD_DIR="$BUDDY_ROOT/build"
PYTHON_PACKAGES_DIR="$BUILD_DIR/python_packages"
FRONTEND_PYTHON_DIR="$BUDDY_ROOT/frontend/Python"

print_info "Starting to sync Python files..."

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    print_error "build directory does not exist, please run cmake configuration first"
    exit 1
fi

# Check if python_packages directory exists
if [ ! -d "$PYTHON_PACKAGES_DIR" ]; then
    print_error "python_packages directory does not exist, please ensure BUDDY_MLIR_ENABLE_PYTHON_PACKAGES is enabled"
    exit 1
fi

# Function to sync Python files
sync_python_files() {
    local src_dir="$1"
    local dest_base="$2"

    # Recursively copy all .py files
    find "$src_dir" -name "*.py" -type f | while read -r file; do
        # Get relative path
        rel_path=$(realpath --relative-to="$src_dir" "$file")
        # Get directory part
        dir_part=$(dirname "$rel_path")
        # Set target directory
        if [ "$dir_part" = "." ]; then
            dest_dir="$dest_base"
        else
            dest_dir="$dest_base/$dir_part"
        fi

        # Create target directory
        mkdir -p "$dest_dir"

        # Copy file
        cp "$file" "$dest_dir/"
        print_info "Synced: $rel_path"
    done
}

# Sync all Python files under frontend/Python
print_info "Syncing frontend/Python files to build/python_packages/buddy/compiler/"
sync_python_files "$FRONTEND_PYTHON_DIR" "$PYTHON_PACKAGES_DIR/buddy/compiler"

print_success "Python files sync completed!"

# Set environment variables
print_info "Setting environment variables..."

export BUDDY_MLIR_BUILD_DIR="$BUILD_DIR"
export LLVM_MLIR_BUILD_DIR="$BUDDY_ROOT/llvm/build"

# Check LLVM build directory
if [ ! -d "$LLVM_MLIR_BUILD_DIR" ]; then
    print_warning "LLVM build directory does not exist: $LLVM_MLIR_BUILD_DIR"
    print_warning "Please ensure LLVM/MLIR is compiled correctly"
fi

# Set PYTHONPATH
export PYTHONPATH="${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}"

print_success "Environment variables setup completed"
print_info "BUDDY_MLIR_BUILD_DIR=$BUDDY_MLIR_BUILD_DIR"
print_info "LLVM_MLIR_BUILD_DIR=$LLVM_MLIR_BUILD_DIR"
print_info "PYTHONPATH updated"
print_success "Script execution completed!"
