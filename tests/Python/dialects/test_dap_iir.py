# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir.dialects import dap, func
from buddy_mlir import ir
from buddy_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def dapIir(dtype, context: ir.Context) -> ir.Module:
    with context, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            memref1d = ir.MemRefType.get(
                [ir.ShapedType.get_dynamic_size()], dtype
            )

            memref2d = ir.MemRefType.get(
                [
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                ],
                dtype,
            )

            @func.FuncOp.from_py_func(memref1d, memref2d, memref1d)
            def buddy_iir(in_, filter, out):
                dap.iir(in_, filter, out)
                return

        return module


# CHECK-LABEL: TEST: testDapIirF32
@run
def testDapIirF32():
    with ir.Context() as context:
        module = dapIir(ir.F32Type.get(), context)
        module.operation.verify()
        # CHECK: dap.iir {{.*}} : memref<?xf32>, memref<?x?xf32>, memref<?xf32>
        print(module)

        pm = PassManager("builtin.module")
        pm.add("lower-dap")
        pm.run(module.operation)

        print(module)


# CHECK-LABEL TEST: testDapIirF64
@run
def testDapIirF64():
    with ir.Context() as context:
        module = dapIir(ir.F64Type.get(), context)
        module.operation.verify()
        # CHECK: dap.iir {{.*}} : memref<?xf64>, memref<?x?xf64>, memref<?xf64>
        print(module)


# CHECK-LABEL: TEST: testDapIirI8
@run
def testDapIirI8():
    with ir.Context() as context:
        module = dapIir(ir.IntegerType.get_signless(8), context)
        module.operation.verify()
        # CHECK: dap.iir {{.*}} : memref<?xi8>, memref<?x?xi8>, memref<?xi8>
        print(module)


# CHECK-LABEL: TEST: testDapIirI16
@run
def testDapIirI16():
    with ir.Context() as context:
        module = dapIir(ir.IntegerType.get_signless(16), context)
        module.operation.verify()
        # CHECK: dap.iir {{.*}} : memref<?xi16>, memref<?x?xi16>, memref<?xi16>
        print(module)


# CHECK-LABEL: TEST: testDapIirI32
@run
def testDapIirI32():
    with ir.Context() as context:
        module = dapIir(ir.IntegerType.get_signless(32), context)
        module.operation.verify()
        # CHECK: dap.iir {{.*}} : memref<?xi32>, memref<?x?xi32>, memref<?xi32>
        print(module)


# CHECK-LABEL: TEST: testDapIirI64
@run
def testDapIirI64():
    with ir.Context() as context:
        module = dapIir(ir.IntegerType.get_signless(64), context)
        module.operation.verify()
        # CHECK: dap.iir {{.*}} : memref<?xi64>, memref<?x?xi64>, memref<?xi64>
        print(module)
