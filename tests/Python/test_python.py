from buddy_mlir.ir import *
from buddy_mlir.passmanager import *
from buddy_mlir.dialects import (
    builtin,
    func,
    linalg,
    gemmini
)

with Context():
    gemmini.register_dialect()
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
    print(str(mod))
    
    pm = PassManager('builtin.module')
    pm.add("convert-linalg-to-gemmini")
    pm.run(mod.operation)
    print(str(mod))