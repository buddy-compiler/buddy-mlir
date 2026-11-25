# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir import ir
from buddy_mlir.dialects import func, xsai_ame


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testTypes
@run
def testTypes():
    with ir.Context(), ir.Location.unknown():
        dyn = ir.ShapedType.get_dynamic_size()
        a = xsai_ame.TileType.get("a", [dyn, 32], "fp16")
        # CHECK: !xsai_ame.tile<a, ?x32, fp16>
        print(a)
        # CHECK: a True 32 fp16
        print(a.kind, a.shape[0] == dyn, a.shape[1], a.element)
        # CHECK: True
        print(ir.Type.parse("!xsai_ame.tile<a, ?x32, fp16>") == a)
        # A parsed type is a TileType (downcast by its TypeID).
        p = ir.Type.parse("!xsai_ame.tile<c, 4x8, i32>")
        # CHECK: True c [4, 8] i32
        print(isinstance(p, xsai_ame.TileType), p.kind, p.shape, p.element)

        s = xsai_ame.SyncType.get()
        # CHECK: !xsai_ame.sync True False
        print(
            s,
            isinstance(ir.Type.parse("!xsai_ame.sync"), xsai_ame.SyncType),
            isinstance(a, xsai_ame.SyncType),
        )

        # The type verifier's diagnostics, as for the builtin checked types.
        for kind, shape, element in (
            ("d", [4, 8], "i8"),
            ("a", [4], "i8"),
            ("a", [0, 8], "i8"),
            ("c", [4, 8], "f64"),
        ):
            try:
                xsai_ame.TileType.get(kind, shape, element)
            except ir.MLIRError as e:
                # CHECK: expected tile kind to be one of a, b, or c
                # CHECK: expected tile shape to have exactly 2 dims
                # CHECK: expected tile dimensions to be positive or dynamic
                # CHECK: unsupported XSAIAME element token: f64
                print(e.error_diagnostics[0].message)


# A dynamic M tile: loads, a cleared accumulator, MMA, store and a sync chain.
# CHECK-LABEL: TEST: testOps
@run
def testOps():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        i8 = ir.IntegerType.get_signless(8)
        i32 = ir.IntegerType.get_signless(32)
        i64 = ir.IntegerType.get_signless(64)
        index = ir.IndexType.get()
        dyn = ir.ShapedType.get_dynamic_size()
        ta = xsai_ame.TileType.get("a", [dyn, 16], "i8")
        tb = xsai_ame.TileType.get("b", [8, 16], "i8")
        tc = xsai_ame.TileType.get("c", [dyn, 8], "i32")
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(
                ir.MemRefType.get([dyn, 16], i8),
                ir.MemRefType.get([16, 8], i8),
                ir.MemRefType.get([dyn, 8], i32),
                i64,
                index,
            )
            def tile_mma(a, b, c, m, target):
                # The !xsai_ame.sync results of msyncregreset and mrelease are
                # inferred.
                sync = xsai_ame.msyncregreset()
                a_tile = xsai_ame.load(ta, a, [m])
                b_tile = xsai_ame.load(tb, b, [], transpose=True)
                acc = xsai_ame.mzero(tc, [m])
                r = xsai_ame.mma(a_tile, b_tile, acc, [m])
                xsai_ame.store(r, c, [m])
                xsai_ame.mfence()
                xsai_ame.macquire(xsai_ame.mrelease(sync), target)

        module.operation.verify()
        # CHECK: func.func @tile_mma(%[[A:.*]]: memref<?x16xi8>, %[[B:.*]]: memref<16x8xi8>, %[[C:.*]]: memref<?x8xi32>, %[[M:.*]]: i64, %[[T:.*]]: index) {
        # CHECK-NEXT: %[[S0:.*]] = xsai_ame.msyncregreset : !xsai_ame.sync
        # CHECK-NEXT: %[[TA:.*]] = xsai_ame.load %[[A]][%[[M]]] : memref<?x16xi8> -> !xsai_ame.tile<a, ?x16, i8>
        # CHECK-NEXT: %[[TB:.*]] = xsai_ame.load %[[B]] {{.*}}transpose = true{{.*}} : memref<16x8xi8> -> !xsai_ame.tile<b, 8x16, i8>
        # CHECK-NEXT: %[[ACC:.*]] = xsai_ame.mzero[%[[M]]] : !xsai_ame.tile<c, ?x8, i32>
        # CHECK-NEXT: %[[R:.*]] = xsai_ame.mma %[[TA]], %[[TB]], %[[ACC]][%[[M]]] : !xsai_ame.tile<a, ?x16, i8>, !xsai_ame.tile<b, 8x16, i8>, !xsai_ame.tile<c, ?x8, i32>{{$}}
        # CHECK-NEXT: xsai_ame.store %[[R]], %[[C]][%[[M]]] : !xsai_ame.tile<c, ?x8, i32>, memref<?x8xi32>{{$}}
        # CHECK-NEXT: xsai_ame.mfence
        # CHECK-NEXT: %[[S1:.*]] = xsai_ame.mrelease %[[S0]] : !xsai_ame.sync
        # CHECK-NEXT: xsai_ame.macquire %[[S1]], %[[T]] : !xsai_ame.sync, index
        print(module)
