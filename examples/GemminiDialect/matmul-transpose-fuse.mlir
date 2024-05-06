// RUN: buddy-opt %s \
// RUN:     --convert-linalg-to-gemmini | \
// RUN: FileCheck %s

func.func @matmul_transpose(%lhs: memref<3x4xi8>, %rhs: memref<4x3xi8>,
                            %output: memref<3x3xi8>) {
    // Matrix-matrix multiplication
    %matmul = memref.alloc() : memref<3x3xi8>
    // CHECK: gemmini.tile_matmul %arg0 %arg1 %arg2 %alloc_0
    linalg.matmul 
        ins(%lhs, %rhs: memref<3x4xi8>, memref<4x3xi8>) 
    outs(%output: memref<3x3xi8>)

    // transpose
    linalg.transpose 
        ins(%matmul: memref<3x3xi8>)
    outs(%output: memref<3x3xi8>)
    permutation = [1, 0]

    memref.dealloc %matmul : memref<3x3xi8>
    return
}
