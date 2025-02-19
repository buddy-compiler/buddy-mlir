# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir.dialects import dap, func
from buddy_mlir import ir


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def dapFir(dtype, context: ir.Context) -> ir.Module:
    with context, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            memref = ir.MemRefType.get(
                [ir.ShapedType.get_dynamic_size()], dtype
            )

            @func.FuncOp.from_py_func(memref, memref, memref)
            def buddy_fir(in_, filter, out):
                dap.fir(in_, filter, out)
                return

        return module


# CHECK-LABEL: TEST: testDapFirF32
@run
def testDapFirF32():
    with ir.Context() as context:
        module = dapFir(ir.F32Type.get(), context)
        module.operation.verify()
        # CHECK: dap.fir {{.*}} : memref<?xf32>, memref<?xf32>, memref<?xf32>
        print(module)


# CHECK-LABEL TEST: testDapFirF64
@run
def testDapFirF64():
    with ir.Context() as context:
        module = dapFir(ir.F64Type.get(), context)
        module.operation.verify()
        # CHECK: dap.fir {{.*}} : memref<?xf64>, memref<?xf64>, memref<?xf64>
        print(module)


# CHECK-LABEL: TEST: testDapFirI8
@run
def testDapFirI8():
    with ir.Context() as context:
        module = dapFir(ir.IntegerType.get_signless(8), context)
        module.operation.verify()
        # CHECK: dap.fir {{.*}} : memref<?xi8>, memref<?xi8>, memref<?xi8>
        print(module)


# CHECK-LABEL: TEST: testDapFirI16
@run
def testDapFirI16():
    with ir.Context() as context:
        module = dapFir(ir.IntegerType.get_signless(16), context)
        module.operation.verify()
        # CHECK: dap.fir {{.*}} : memref<?xi16>, memref<?xi16>, memref<?xi16>
        print(module)


# CHECK-LABEL: TEST: testDapFirI32
@run
def testDapFirI32():
    with ir.Context() as context:
        module = dapFir(ir.IntegerType.get_signless(32), context)
        module.operation.verify()
        # CHECK: dap.fir {{.*}} : memref<?xi32>, memref<?xi32>, memref<?xi32>
        print(module)


# CHECK-LABEL: TEST: testDapFirI64
@run
def testDapFirI64():
    with ir.Context() as context:
        module = dapFir(ir.IntegerType.get_signless(64), context)
        module.operation.verify()
        # CHECK: dap.fir {{.*}} : memref<?xi64>, memref<?xi64>, memref<?xi64>
        print(module)
