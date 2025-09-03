# RUN: %PYTHON %s 2>&1 | FileCheck %s

from buddy_mlir import ir
from buddy_mlir.dialects import arith, linalg, memref, func
from buddy_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL TEST: testLinalgConv2DConversion
@run
def testLinalgConv2DConversion():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        f32 = ir.F32Type.get()

        memref2x2x5x5 = ir.MemRefType.get([2, 2, 5, 5], f32)
        memref2x2x3x3 = ir.MemRefType.get([2, 2, 3, 3], f32)

        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(memref2x2x5x5, memref2x2x3x3)
            def linalg_conv2d(input, weight):
                mem2 = memref.alloc(memref2x2x3x3, [], [])
                linalg.conv_2d_nchw_fchw(input, weight, outs=[mem2])

                return

        module.operation.verify()

        pm = PassManager("builtin.module")
        pm.add("convert-linalg-to-gemmini")
        pm.run(module.operation)
        # CHECK: gemmini.tile_conv {{.+}} :
        # CHECK-SAME: memref<2x5x5x2xf32> memref<18x2xf32> memref<2xi32> memref<18x2xf32> i64 i64
        print(module)
