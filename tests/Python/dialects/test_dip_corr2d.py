# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir.dialects import dip, func
from buddy_mlir import ir


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def dipCorr2D(
    dtype, boundary_option: ir.Attribute, context: ir.Context
) -> ir.Module:
    with context, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index = ir.IndexType.get()
            memref = ir.MemRefType.get(
                [
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                ],
                dtype,
            )

            @func.FuncOp.from_py_func(
                memref, memref, memref, index, index, dtype
            )
            def buddy_corr2d(
                input, identity, output, kernelAnchorX, kernelAnchorY, c
            ):
                dip.corr_2d(
                    input,
                    identity,
                    output,
                    kernelAnchorX,
                    kernelAnchorY,
                    c,
                    boundary_option,
                )

        return module


# CHECK-LABEL: TEST: testDipCorr2DConstantPaddingF32
@run
def testDipCorr2DConstantPaddingF32():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.F32Type.get(),
            ir.Attribute.parse("#dip<boundary_option <CONSTANT_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
        print(module)


# CHECK-LABEL: TEST: testDipCorr2DConstantPaddingF64
@run
def testDipCorr2DConstantPaddingF64():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.F64Type.get(),
            ir.Attribute.parse("#dip<boundary_option <CONSTANT_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, index, index, f64
        print(module)


# CHECK-LABEL: TEST: testDipCorr2DConstantPaddingI8
@run
def testDipCorr2DConstantPaddingI8():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.IntegerType.get_signless(8),
            ir.Attribute.parse("#dip<boundary_option <CONSTANT_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi8>, index, index, i8
        print(module)


# CHECK-LABEL: TEST: testDipCorr2DConstantPaddingI32
@run
def testDipCorr2DConstantPaddingI32():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.IntegerType.get_signless(32),
            ir.Attribute.parse("#dip<boundary_option <CONSTANT_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <CONSTANT_PADDING>{{.*}} : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, index, index, i32
        print(module)


# CHECK-LABEL: TEST: testDipCorr2DConstantPaddingI64
@run
def testDipCorr2DConstantPaddingI64():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.IntegerType.get_signless(64),
            ir.Attribute.parse("#dip<boundary_option <CONSTANT_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <CONSTANT_PADDING{{.*}} : memref<?x?xi64>, memref<?x?xi64>, memref<?x?xi64>, index, index, i64
        print(module)


# CHECK-LABEL: TEST: testDipCorr2DReplicatePaddingF32
@run
def testDipCorr2DReplicatePaddingF32():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.F32Type.get(),
            ir.Attribute.parse("#dip<boundary_option <REPLICATE_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
        print(module)


# CHECK-LABEL: TEST: testDipCorr2DReplicatePaddingF64
@run
def testDipCorr2DReplicatePaddingF64():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.F64Type.get(),
            ir.Attribute.parse("#dip<boundary_option <REPLICATE_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, index, index, f64
        print(module)


# CHECK-LABEL: TEST: testDipCorr2DReplicatePaddingI8
@run
def testDipCorr2DReplicatePaddingI8():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.IntegerType.get_signless(8),
            ir.Attribute.parse("#dip<boundary_option <REPLICATE_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi8>, index, index, i8
        print(module)


# CHECK-LABEL: TEST: testDipCorr2DReplicatePaddingI32
@run
def testDipCorr2DReplicatePaddingI32():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.IntegerType.get_signless(32),
            ir.Attribute.parse("#dip<boundary_option <REPLICATE_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, index, index, i32
        print(module)


# CHECK-LABEL: TEST: testDipCorr2DReplicatePaddingI64
@run
def testDipCorr2DReplicatePaddingI64():
    with ir.Context() as context:
        module = dipCorr2D(
            ir.IntegerType.get_signless(64),
            ir.Attribute.parse("#dip<boundary_option <REPLICATE_PADDING>>"),
            context,
        )
        module.operation.verify()
        # CHECK: dip.corr_2d <REPLICATE_PADDING>{{.*}} : memref<?x?xi64>, memref<?x?xi64>, memref<?x?xi64>, index, index, i64
        print(module)
