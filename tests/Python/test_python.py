# RUN: %PYTHON %s 2>&1 | FileCheck %s

from buddy_mlir.ir import *
from buddy_mlir.passmanager import *
from buddy_mlir.dialects import (
    builtin,
    func,
    linalg,
    gemmini
)

with Context():
    mod = Module.parse("""
        %0 = arith.constant 0 : i8
        %1 = arith.constant 1 : i8
        %2 = arith.constant 2 : i8
        %mem0 = memref.alloc() : memref<8x8xi8>
        %mem1 = memref.alloc() : memref<8x8xi8>
        %mem2 = memref.alloc() : memref<8x8xi8>
        linalg.fill
            ins(%2 : i8)
        outs(%mem0 : memref<8x8xi8>)
        linalg.fill
            ins(%1 : i8)
        outs(%mem1 : memref<8x8xi8>)
        linalg.matmul
            ins(%mem0, %mem1 : memref<8x8xi8>, memref<8x8xi8>)
        outs(%mem2 : memref<8x8xi8>)
        gemmini.print %mem2 : memref<8x8xi8>
        """)
    
    pm = PassManager('builtin.module')
    pm.add("convert-linalg-to-gemmini")
    pm.run(mod.operation)
    
    # CHECK: module {
    # CHECK: %c0_i8 = arith.constant 0 : i8
    # CHECK: %c1_i8 = arith.constant 1 : i8
    # CHECK: %c2_i8 = arith.constant 2 : i8
    # CHECK: %alloc = memref.alloc() : memref<8x8xi8>
    # CHECK: %alloc_0 = memref.alloc() : memref<8x8xi8>
    # CHECK: %alloc_1 = memref.alloc() : memref<8x8xi8>
    # CHECK: linalg.fill ins(%c2_i8 : i8) outs(%alloc : memref<8x8xi8>)
    # CHECK: linalg.fill ins(%c1_i8 : i8) outs(%alloc_0 : memref<8x8xi8>)
    # CHECK: %alloc_2 = memref.alloc() : memref<8x8xi32>
    # CHECK: %c0_i32 = arith.constant 0 : i32
    # CHECK: linalg.fill ins(%c0_i32 : i32) outs(%alloc_2 : memref<8x8xi32>)
    # CHECK: gemmini.tile_matmul %alloc %alloc_0 %alloc_1 %alloc_2 : memref<8x8xi8> memref<8x8xi8> memref<8x8xi8> memref<8x8xi32>
    # CHECK: memref.dealloc %alloc_2 : memref<8x8xi32>
    # CHECK: gemmini.print %alloc_1 : memref<8x8xi8>
    # CHECK: }
    print(str(mod))
