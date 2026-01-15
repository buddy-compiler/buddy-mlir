#!/bin/bash
# Build script for IME print tests
#
# Usage: ./build_all_tests.sh
# 
# Prerequisites:
# - buddy-mlir built in ../../build
# - RISCV toolchain with riscv64-unknown-linux-gnu-gcc in PATH
#   or set SPACEMIT_GCC environment variable

BUDDY_OPT=../../build/bin/buddy-opt
BUDDY_TRANSLATE=../../build/bin/buddy-translate
BUDDY_LLC=../../build/bin/buddy-llc
SPACEMIT_GCC=${SPACEMIT_GCC:-riscv64-unknown-linux-gnu-gcc}

PASSES="-lower-ime -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts"

build_test() {
    local name=$1
    local mlir_file=$2
    local runtime_file=$3
    
    echo "Building $name..."
    
    # Generate assembly
    $BUDDY_OPT $mlir_file $PASSES | \
    $BUDDY_TRANSLATE -buddy-to-llvmir | \
    $BUDDY_LLC -filetype=asm -mtriple=riscv64 -mattr=+m,+v,+buddyext -o ${name}.s
    
    if [ $? -ne 0 ]; then
        echo "Failed to generate assembly for $name"
        return 1
    fi
    
    # Compile to executable
    $SPACEMIT_GCC -march=rv64gcv -static ${name}.s $runtime_file -o ${name}_test
    
    if [ $? -ne 0 ]; then
        echo "Failed to compile $name"
        return 1
    fi
    
    echo "$name built successfully: ${name}_test"
}

# Build vmadot (signed x signed)
build_test "vmadot" "vmadot_print_test.mlir" "runtime_vmadot.c"

# Build vmadotu (unsigned x unsigned)  
build_test "vmadotu" "vmadotu_print_test.mlir" "runtime_vmadotu.c"

# Build vmadotsu (signed x unsigned)
build_test "vmadotsu" "vmadotsu_print_test.mlir" "runtime_vmadotsu.c"

# Build vmadotus (unsigned x signed)
build_test "vmadotus" "vmadotus_print_test.mlir" "runtime_vmadotus.c"

# Build vfmadot (floating-point) - uses different print function
# build_test "vfmadot" "vfmadot_print_test.mlir" "runtime_vfmadot.c"

# Build vmadotn (signed x signed, dynamic slide)
build_test "vmadotn" "vmadotn_print_test.mlir" "runtime_vmadotn.c"

# Build vmadotnu (unsigned x unsigned, dynamic slide)
build_test "vmadotnu" "vmadotnu_print_test.mlir" "runtime_vmadotnu.c"

# Build vmadotnsu (signed x unsigned, dynamic slide)
build_test "vmadotnsu" "vmadotnsu_print_test.mlir" "runtime_vmadotnsu.c"

# Build vmadotnus (unsigned x signed, dynamic slide)
build_test "vmadotnus" "vmadotnus_print_test.mlir" "runtime_vmadotnus.c"

# Build vfmadotn (floating-point, dynamic slide) - uses different print function
# build_test "vfmadotn" "vfmadotn_print_test.mlir" "runtime_vfmadotn.c"

echo ""
echo "All builds complete. Executables:"
ls -la *_test 2>/dev/null || echo "No test executables found"
