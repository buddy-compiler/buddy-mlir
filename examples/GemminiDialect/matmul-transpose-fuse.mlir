// RUN: buddy-opt %s \
// RUN:     --convert-linalg-to-gemmini | \
// RUN: FileCheck %s

memref.global "private" @gv1 : memref<3x4xi8> = dense<[[1, 2, 3, 4],
                                                       [5, 6, 7, 8],
                                                       [9, 10, 11, 12]]>
memref.global "private" @gv2 : memref<4x3xi8> = dense<[[1, 1, 1],
                                                       [1, 1, 1],
                                                       [1, 1, 1],
                                                       [1, 1, 1]]>

func.func @main() -> i8 {
    %arrayA = memref.get_global @gv1 : memref<3x4xi8>
    %arrayB = memref.get_global @gv2 : memref<4x3xi8>
    %arrayC = memref.alloc() : memref<3x3xi8>
    %cst0 = arith.constant 0 : i8
    gemmini.print %arrayC : memref<3x3xi8>
    // Matrix-matrix multiplication
    // CHECK: gemmini.tile_matmul %1 %0 %alloc %alloc_0 {aTranspose = true, bTranspose = true} : 
    // CHECK-SAME: memref<4x3xi8> memref<3x4xi8> memref<3x3xi8> memref<3x4xi32>
    linalg.matmul 
        ins(%arrayA, %arrayB: memref<3x4xi8>, memref<4x3xi8>) 
    outs(%arrayC: memref<3x3xi8>)

    // transpose
    linalg.transpose 
        ins(%arrayC: memref<3x3xi8>)
    outs(%arrayC: memref<3x3xi8>)
    permutation = [1, 0]

    gemmini.print %arrayC : memref<3x3xi8>
    memref.dealloc %arrayC : memref<3x3xi8>

    return %cst0 : i8
}
