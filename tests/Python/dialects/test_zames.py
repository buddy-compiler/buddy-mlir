# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir import ir
from buddy_mlir.dialects import func, zames


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testTypes
@run
def testTypes():
    with ir.Context(), ir.Location.unknown():
        dyn = ir.ShapedType.get_dynamic_size()
        a = zames.TileType.get("a", [dyn, 32], "fp16")
        # CHECK: !zames.tile<a, ?x32, fp16>
        print(a)
        # CHECK: a True 32 fp16
        print(a.kind, a.shape[0] == dyn, a.shape[1], a.element)
        # CHECK: True
        print(ir.Type.parse("!zames.tile<a, ?x32, fp16>") == a)
        # A parsed type is a TileType (downcast by its TypeID).
        p = ir.Type.parse("!zames.tile<c, 4x8, i32>")
        # CHECK: True c [4, 8] i32
        print(isinstance(p, zames.TileType), p.kind, p.shape, p.element)

        s = zames.SyncType.get()
        # CHECK: !zames.sync True False
        print(
            s,
            isinstance(ir.Type.parse("!zames.sync"), zames.SyncType),
            isinstance(a, zames.SyncType),
        )

        # The type verifier's diagnostics, as for the builtin checked types.
        for kind, shape, element in (
            ("d", [4, 8], "i8"),
            ("a", [4], "i8"),
            ("a", [0, 8], "i8"),
            ("c", [4, 8], "f64"),
        ):
            try:
                zames.TileType.get(kind, shape, element)
            except ir.MLIRError as e:
                # CHECK: expected tile kind to be one of a, b, or c
                # CHECK: expected tile shape to have exactly 2 dims
                # CHECK: expected tile dimensions to be positive or dynamic
                # CHECK: unsupported ZAMES element token: f64
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
        ta = zames.TileType.get("a", [dyn, 16], "i8")
        tb = zames.TileType.get("b", [8, 16], "i8")
        tc = zames.TileType.get("c", [dyn, 8], "i32")
        with ir.InsertionPoint(module.body):

            @func.FuncOp.from_py_func(
                ir.MemRefType.get([dyn, 16], i8),
                ir.MemRefType.get([16, 8], i8),
                ir.MemRefType.get([dyn, 8], i32),
                i64,
                index,
            )
            def tile_mma(a, b, c, m, target):
                # The !zames.sync results of msyncregreset and mrelease are
                # inferred.
                sync = zames.msyncregreset()
                a_tile = zames.load(ta, a, [m])
                b_tile = zames.load(tb, b, [], transpose=True)
                acc = zames.mzero(tc, [m])
                r = zames.mma(a_tile, b_tile, acc, [m])
                zames.store(r, c, [m])
                zames.mfence()
                zames.macquire(zames.mrelease(sync), target)

        module.operation.verify()
        # CHECK: func.func @tile_mma(%[[A:.*]]: memref<?x16xi8>, %[[B:.*]]: memref<16x8xi8>, %[[C:.*]]: memref<?x8xi32>, %[[M:.*]]: i64, %[[T:.*]]: index) {
        # CHECK-NEXT: %[[S0:.*]] = zames.msyncregreset : !zames.sync
        # CHECK-NEXT: %[[TA:.*]] = zames.load %[[A]][%[[M]]] : memref<?x16xi8> -> !zames.tile<a, ?x16, i8>
        # CHECK-NEXT: %[[TB:.*]] = zames.load %[[B]] {{.*}}transpose = true{{.*}} : memref<16x8xi8> -> !zames.tile<b, 8x16, i8>
        # CHECK-NEXT: %[[ACC:.*]] = zames.mzero[%[[M]]] : !zames.tile<c, ?x8, i32>
        # CHECK-NEXT: %[[R:.*]] = zames.mma %[[TA]], %[[TB]], %[[ACC]][%[[M]]] : !zames.tile<a, ?x16, i8>, !zames.tile<b, 8x16, i8>, !zames.tile<c, ?x8, i32>{{$}}
        # CHECK-NEXT: zames.store %[[R]], %[[C]][%[[M]]] : !zames.tile<c, ?x8, i32>, memref<?x8xi32>{{$}}
        # CHECK-NEXT: zames.mfence
        # CHECK-NEXT: %[[S1:.*]] = zames.mrelease %[[S0]] : !zames.sync
        # CHECK-NEXT: zames.macquire %[[S1]], %[[T]] : !zames.sync, index
        print(module)
