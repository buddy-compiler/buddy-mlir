# RUN: %PYTHON %s 2>&1 | FileCheck %s

from buddy_mlir import ir
from buddy_mlir.dialects import arith, linalg, memref, func
from buddy_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testLinalgMatmulConversion
@run
def testLinalgMatmulConversion():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        i8 = ir.IntegerType.get_signless(8)

        memref8x8 = ir.MemRefType.get([8, 8], i8)

        with ir.InsertionPoint(module.body):
            c1 = arith.ConstantOp(i8, 1)
            c2 = arith.ConstantOp(i8, 2)

            mem0 = memref.alloc(memref8x8, [], [])
            mem1 = memref.alloc(memref8x8, [], [])
            mem2 = memref.alloc(memref8x8, [], [])

            linalg.fill(c2, outs=[mem0])
            linalg.fill(c1, outs=[mem1])
            linalg.matmul(mem0, mem1, outs=[mem2])

        module.operation.verify()

        pm = PassManager("builtin.module")
        pm.add("convert-linalg-to-gemmini")
        pm.run(module.operation)

        # CHECK: gemmini.tile_matmul [[MEM0:%.*]] [[MEM1:%.*]] [[MEM2:%.*]] :
        # CHECK-SAME: memref<8x8xi8> memref<8x8xi8> memref<8x8xi8> memref<8x8xi32>
        print(module)


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
