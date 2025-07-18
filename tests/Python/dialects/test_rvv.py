# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir.dialects import arith, func, rvv
from buddy_mlir import ir
from buddy_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testRVVLegalizeForLLVM
@run
def testRVVLegalizeForLLVM():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index_type = ir.IndexType.get()

            @func.FuncOp.from_py_func(index_type, results=[index_type])
            def rvv_setvl(avl):
                # SEW = 32
                sew = arith.ConstantOp(index_type, 2)
                # LMUL = 2
                lmul = arith.ConstantOp(index_type, 2)
                vl = rvv.setvl(index_type, avl, sew, lmul)
                func.return_([vl])
                return

        module.operation.verify()

        # CHECK:      func @rvv_setvl(%[[ARG0:.*]]: index) -> index {
        # CHECK-NEXT:   %[[SEW:.*]] = arith.constant 2 : index
        # CHECK-NEXT:   %[[LMUL:.*]] = arith.constant 2 : index
        # CHECK-NEXT:   rvv.setvl %[[ARG0]], %[[SEW]], %[[LMUL]] : index
        # CHECK-NEXT:   return %[[RESULT:.*]] : index
        # CHECK-NEXT: }
        print(module)

        pm = PassManager("builtin.module")
        pm.add("lower-rvv")
        pm.run(module.operation)
        # CHECK: rvv.intr.vsetvli{{.*}} : (i64, i64, i64) -> i64
        print(module)


# CHECK-LABEL: TEST: testRVVRsqrtLegalizeForLLVM
@run
def testRVVRsqrtLegalizeForLLVM():
    with ir.Context():
        module = ir.Module.parse(
            """
            func.func @rvv_rsqrt(%arg0: memref<?xf32>) {
                %c0 = arith.constant 0 : index

                %sew = arith.constant 2 : index
                %lmul = arith.constant 1 : index
                %avl8 = arith.constant 8 : index
                %vl8 = rvv.setvl %avl8, %sew, %lmul : index

                %load_vec = rvv.load %arg0[%c0], %vl8 : memref<?xf32>, vector<[4]xf32>, index
                %rsqrt_vec = math.rsqrt %load_vec : vector<[4]xf32>
                rvv.store %rsqrt_vec, %arg0[%c0], %vl8 : vector<[4]xf32>, memref<?xf32>, index

                return
            }
            """
        )

        module.operation.verify()

        pm = PassManager("builtin.module")
        pm.add("lower-rvv")
        pm.run(module.operation)

        # CHECK: rvv.intr.vsetvli{{.*}} : (i64, i64, i64) -> i64
        # CHECK: rvv.intr.vle{{.*}} : (vector<[4]xf32>, !llvm.ptr, i64) -> vector<[4]xf32>
        # CHECK: rvv.intr.vse{{.*}} : (vector<[4]xf32>, !llvm.ptr, i64) -> ()
        print(module)
