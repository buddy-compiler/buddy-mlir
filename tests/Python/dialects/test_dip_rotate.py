# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir.dialects import dip, func
from buddy_mlir import ir


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def dipRotate2D(dtype, context: ir.Context) -> ir.Module:
    with context, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get()
            memref = ir.MemRefType.get(
                [
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                ],
                dtype,
            )

            @func.FuncOp.from_py_func(memref, f32, memref)
            def buddy_resize2d(input, angle, output):
                dip.rotate_2d(input, angle, output)

        return module


# CHECK-LABEL: TEST: testDipRotate2DF32
@run
def testDipRotate2DF32():
    with ir.Context() as context:
        module = dipRotate2D(ir.F32Type.get(), context)
        module.operation.verify()
        # CHECK: dip.rotate_2d {{.*}} : memref<?x?xf32>, f32, memref<?x?xf32>
        print(module)


# CHECK-LABEL: TEST: testDipRotate2DF64
@run
def testDipRotate2DF64():
    with ir.Context() as context:
        module = dipRotate2D(ir.F64Type.get(), context)
        module.operation.verify()
        # CHECK: dip.rotate_2d {{.*}} : memref<?x?xf64>, f32, memref<?x?xf64>
        print(module)


# CHECK-LABEL: TEST: testDipRotate2DI8
@run
def testDipRotate2DI8():
    with ir.Context() as context:
        module = dipRotate2D(ir.IntegerType.get_signless(8), context)
        module.operation.verify()
        # CHECK: dip.rotate_2d {{.*}} : memref<?x?xi8>, f32, memref<?x?xi8>
        print(module)


# CHECK-LABEL: TEST: testDipRotate2DI32
@run
def testDipRotate2DI32():
    with ir.Context() as context:
        module = dipRotate2D(ir.IntegerType.get_signless(32), context)
        module.operation.verify()
        # CHECK: dip.rotate_2d {{.*}} : memref<?x?xi32>, f32, memref<?x?xi32>
        print(module)


# CHECK-LABEL: TEST: testDipRotate2DI64
@run
def testDipRotate2DI64():
    with ir.Context() as context:
        module = dipRotate2D(ir.IntegerType.get_signless(64), context)
        module.operation.verify()
        # CHECK: dip.rotate_2d {{.*}} : memref<?x?xi64>, f32, memref<?x?xi64>
        print(module)
