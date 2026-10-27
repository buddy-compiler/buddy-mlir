# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir import ir
from buddy_mlir.dialects import arith, func, vector, vir
from buddy_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testDynamicVectorType
@run
def testDynamicVectorType():
    with ir.Context():
        dyn = ir.ShapedType.get_dynamic_size()
        f32 = ir.F32Type.get()
        t = vir.DynamicVectorType.get([2, dyn], f32)
        # CHECK: !vir.vec<2x?xf32>
        print(t)
        # CHECK: [2, {{-?[0-9]+}}] f32 2 True None
        print(
            t.shape,
            t.element_type,
            t.rank,
            t.is_dynamic_dim(1),
            t.scaling_factor,
        )
        # CHECK: True
        print(ir.Type.parse("!vir.vec<2x?xf32>") == t)

        m2 = vir.ScalingFactorAttr.get("m", 2)
        s = vir.DynamicVectorType.get(
            [dyn], ir.IntegerType.get_signless(8), scaling_factor=m2
        )
        # CHECK: !vir.vec<?xi8, #vir.sf<m2>>
        print(s)
        # CHECK: True
        print(ir.Type.parse("!vir.vec<?xi8, m2>") == s)
        # CHECK: m 2
        print(s.scaling_factor.ratio, s.scaling_factor.value)

        # A parsed type is a DynamicVectorType (downcast by its TypeID).
        p = ir.Type.parse("!vir.vec<?xindex>")
        # CHECK: True index
        print(isinstance(p, vir.DynamicVectorType), p.element_type)

        for shape, elem in (
            ([2], f32),
            ([dyn, dyn], f32),
            ([2, dyn], ir.NoneType.get()),
        ):
            try:
                vir.DynamicVectorType.get(shape, elem)
            except ValueError as e:
                # CHECK: the last dim of a dynamic vector is the dynamic one
                # CHECK: the dims before the last one of a dynamic vector are static
                # CHECK: the element type of a dynamic vector is an integer
                print(e)
        try:
            vir.ScalingFactorAttr.get("m", 3)
        except ValueError as e:
            # CHECK: the value of a scaling factor is 2, 4, 8, 16, 32 or 64
            print(e)
        try:
            other = vir.ScalingFactorAttr.get("m", 2, context=ir.Context())
            vir.DynamicVectorType.get([dyn], f32, scaling_factor=other)
        except ValueError as e:
            # CHECK: the scaling factor and the element type must be in the same context
            print(e)


# The query of a head of 128 as two rows: RoPE (rotate_half is a shuffle of
# the rows of -q and q) and a dot product with a row of keys.
# CHECK-LABEL: TEST: testRoPEDot
@run
def testRoPEDot():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        f32 = ir.F32Type.get()
        index = ir.IndexType.get()
        rows = ir.MemRefType.get([2, 64], f32)
        vec = vir.DynamicVectorType.get(
            [2, ir.ShapedType.get_dynamic_size()], f32
        )
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(rows, rows, rows, rows, results=[f32])
            def rope_dot(q, cos, sin, k):
                vl = arith.ConstantOp(index, 64)
                zero = arith.ConstantOp(f32, 0.0)
                op = vir.SetVLOp(vl, [f32])
                with ir.InsertionPoint(op.body):
                    x = vir.load(vec, q, [])
                    c = vir.load(vec, cos, [])
                    s = vir.load(vec, sin, [])
                    rot = vir.shuffle(arith.negf(x), x, [1, 2])
                    y = vir.fma(x, c, arith.mulf(rot, s))
                    kk = vir.load(vec, k, [])
                    d = vir.reduce(
                        f32, arith.mulf(y, kk), zero, "add", reassoc=True
                    )
                    vector.YieldOp([d])
                # The result types of the ops are DynamicVectorType.
                assert isinstance(x.type, vir.DynamicVectorType)
                func.ReturnOp([op.result])

        module.operation.verify()
        # CHECK:      func.func @rope_dot(%[[Q:.*]]: memref<2x64xf32>, %[[C:.*]]: memref<2x64xf32>, %[[S:.*]]: memref<2x64xf32>, %[[K:.*]]: memref<2x64xf32>) -> f32 {
        # CHECK:        %[[R:.*]] = vir.set_vl %{{.*}} : index {
        # CHECK-NEXT:     %[[X:.*]] = vir.load %[[Q]][] : memref<2x64xf32> -> !vir.vec<2x?xf32>
        # CHECK:          %[[NX:.*]] = arith.negf %[[X]] : !vir.vec<2x?xf32>
        # CHECK-NEXT:     %[[ROT:.*]] = vir.shuffle %[[NX]], %[[X]] [1, 2] : !vir.vec<2x?xf32>
        # CHECK:          vir.fma %[[X]], {{.*}} : !vir.vec<2x?xf32>
        # CHECK:          %[[D:.*]] = vir.reduce {{.*}} {kind = "add", reassoc} : !vir.vec<2x?xf32>, f32 -> f32
        # CHECK-NEXT:     vector.yield %[[D]] : f32
        # CHECK-NEXT:   } -> f32
        # CHECK-NEXT:   return %[[R]] : f32
        print(module)

        # The lowering is registered for Python.
        pm = PassManager.parse(
            "builtin.module(lower-vir-to-vector{rvv-vlen=1024})"
        )
        pm.run(module.operation)
        # CHECK-NOT: vir.
        # CHECK: vector.shuffle {{.*}} : vector<128xf32>, vector<128xf32>
        # CHECK: vector.reduction <add>, {{.*}} fastmath<reassoc> : vector<128xf32> into f32
        print(module)


# A region without results: a strip-mined loop over vl.
# CHECK-LABEL: TEST: testScale
@run
def testScale():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        f32 = ir.F32Type.get()
        index = ir.IndexType.get()
        buf = ir.MemRefType.get([ir.ShapedType.get_dynamic_size()], f32)
        vec = vir.DynamicVectorType.get([ir.ShapedType.get_dynamic_size()], f32)
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(buf, index, f32)
            def scale(a, n, w):
                op = vir.SetVLOp(n)
                with ir.InsertionPoint(op.body):
                    x = vir.load(vec, a, [])
                    y = arith.mulf(x, vir.broadcast(vec, w))
                    vir.store(y, a, [])
                    vector.YieldOp([])

        module.operation.verify()
        # CHECK: vir.set_vl %{{.*}} : index {
        # CHECK:   vir.broadcast %{{.*}} : f32 -> !vir.vec<?xf32>
        # CHECK:   vir.store %{{.*}}, %{{.*}}[] : !vir.vec<?xf32> -> memref<?xf32>
        # CHECK:   vector.yield
        # CHECK: }
        print(module)

        pm = PassManager.parse(
            "builtin.module(lower-vir-to-vector{use-scalable=true rvv-vlen=256})"
        )
        pm.run(module.operation)
        # CHECK: scf.for {{.*}} step %{{.*}} {
        # CHECK:   vector.load {{.*}} : memref<?xf32>, vector<[16]xf32>
        print(module)
