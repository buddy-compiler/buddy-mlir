# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir.dialects import dip, func
from buddy_mlir import ir


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


def dipResize2D(
    dtype, interpolation_attr: ir.Attribute, context: ir.Context
) -> ir.Module:
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

            @func.FuncOp.from_py_func(memref, f32, f32, memref)
            def buddy_resize2d(
                input,
                horizontal_scaling_factor,
                vertical_scaling_factor,
                output,
            ):
                dip.resize_2d(
                    input,
                    horizontal_scaling_factor,
                    vertical_scaling_factor,
                    output,
                    interpolation_attr,
                )

        return module


def dipResize4DNchw(
    dtype,
    interpolation_attr: ir.Attribute,
    context: ir.Context,
) -> ir.Module:
    with context, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get()
            memref = ir.MemRefType.get(
                [
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                ],
                dtype,
            )

            @func.FuncOp.from_py_func(memref, f32, f32, memref)
            def buddy_resize4d(
                input,
                horizontal_scaling_factor,
                vertical_scaling_factor,
                output,
            ):
                dip.resize_4d_nchw(
                    input,
                    horizontal_scaling_factor,
                    vertical_scaling_factor,
                    output,
                    interpolation_attr,
                )

        return module


def dipResize4DNhwc(
    dtype,
    interpolation_attr: ir.Attribute,
    context: ir.Context,
) -> ir.Module:
    with context, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f32 = ir.F32Type.get()
            memref = ir.MemRefType.get(
                [
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                ],
                dtype,
            )

            @func.FuncOp.from_py_func(memref, f32, f32, memref)
            def buddy_resize4d(
                input,
                horizontal_scaling_factor,
                vertical_scaling_factor,
                output,
            ):
                dip.resize_4d_nhwc(
                    input,
                    horizontal_scaling_factor,
                    vertical_scaling_factor,
                    output,
                    interpolation_attr,
                )

        return module


# CHECK-LABEL: TEST: testDipResize2DNearestInpterpolationF32
@run
def testDipResize2DNearestInpterpolationF32():
    with ir.Context() as context:
        module = dipResize2D(
            ir.F32Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type NEAREST_NEIGHBOUR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?xf32>, f32, f32, memref<?x?xf32>
        print(module)


# CHECK-LABEL: TEST: testDipResize2DNearestInpterpolationF64
@run
def testDipResize2DNearestInpterpolationF64():
    with ir.Context() as context:
        module = dipResize2D(
            ir.F64Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type NEAREST_NEIGHBOUR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?xf64>, f32, f32, memref<?x?xf64>
        print(module)


# CHECK-LABEL: TEST: testDipResize2DBilinearF32
@run
def testDipResize2DBilinearF32():
    with ir.Context() as context:
        module = dipResize2D(
            ir.F32Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type BILINEAR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_2d BILINEAR_INTERPOLATION{{.*}} : memref<?x?xf32>, f32, f32, memref<?x?xf32>
        print(module)


# CHECK-LABEL: TEST: testDipResize2DBilinearF64
@run
def testDipResize2DBilinearF64():
    with ir.Context() as context:
        module = dipResize2D(
            ir.F64Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type BILINEAR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_2d BILINEAR_INTERPOLATION{{.*}} : memref<?x?xf64>, f32, f32, memref<?x?xf64>
        print(module)


# CHECK-LABEL: TEST: testDipResize4DNchwNearestInterpolationF32
@run
def testDipResize4DNchwNearestInterpolationF32():
    with ir.Context() as context:
        module = dipResize4DNchw(
            ir.F32Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type NEAREST_NEIGHBOUR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_4d_nchw NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
        print(module)


# CHECK-LABEL: TEST: testDipResize4DNchwNearestInterpolationF64
@run
def testDipResize4DNchwNearestInterpolationF64():
    with ir.Context() as context:
        module = dipResize4DNchw(
            ir.F64Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type NEAREST_NEIGHBOUR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_4d_nchw NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?x?x?xf64>, f32, f32, memref<?x?x?x?xf64>
        print(module)


# CHECK-LABEL: TEST: testDipResize4DNchwBilinearF32
@run
def testDipResize4DNchwBilinearF32():
    with ir.Context() as context:
        module = dipResize4DNchw(
            ir.F32Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type BILINEAR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_4d_nchw BILINEAR_INTERPOLATION{{.*}} : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
        print(module)


# CHECK-LABEL: TEST: testDipResize4DNchwBilinearF64
@run
def testDipResize4DNchwBilinearF64():
    with ir.Context() as context:
        module = dipResize4DNchw(
            ir.F64Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type BILINEAR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_4d_nchw BILINEAR_INTERPOLATION{{.*}} : memref<?x?x?x?xf64>, f32, f32, memref<?x?x?x?xf64>
        print(module)


# CHECK-LABEL: TEST: testDipResize4DNhwcNearestInterpolationF32
@run
def testDipResize4DNhwcNearestInterpolationF32():
    with ir.Context() as context:
        module = dipResize4DNhwc(
            ir.F32Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type NEAREST_NEIGHBOUR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_4d_nhwc NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
        print(module)


# CHECK-LABEL: TEST: testDipResize4DNhwcNearestInterpolationF64
@run
def testDipResize4DNhwcNearestInterpolationF64():
    with ir.Context() as context:
        module = dipResize4DNhwc(
            ir.F64Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type NEAREST_NEIGHBOUR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_4d_nhwc NEAREST_NEIGHBOUR_INTERPOLATION{{.*}} : memref<?x?x?x?xf64>, f32, f32, memref<?x?x?x?xf64>
        print(module)


# CHECK-LABEL: TEST: testDipResize4DNhwcBilinearF32
@run
def testDipResize4DNhwcBilinearF32():
    with ir.Context() as context:
        module = dipResize4DNhwc(
            ir.F32Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type BILINEAR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_4d_nhwc BILINEAR_INTERPOLATION{{.*}} : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
        print(module)


# CHECK-LABEL: TEST: testDipResize4DNhwcBilinearF64
@run
def testDipResize4DNhwcBilinearF64():
    with ir.Context() as context:
        module = dipResize4DNhwc(
            ir.F64Type.get(),
            ir.Attribute.parse(
                "#dip<interpolation_type BILINEAR_INTERPOLATION>"
            ),
            context,
        )
        module.operation.verify()
        # CHECK: dip.resize_4d_nhwc BILINEAR_INTERPOLATION{{.*}} : memref<?x?x?x?xf64>, f32, f32, memref<?x?x?x?xf64>
        print(module)
