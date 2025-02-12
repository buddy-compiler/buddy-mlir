# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir.dialects import func, linalg
from buddy_mlir import ir
from buddy_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testMatmulVectorize
@run
def testMatmulVectorize():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            memref = ir.MemRefType.get(
                [
                    ir.ShapedType.get_dynamic_size(),
                    ir.ShapedType.get_dynamic_size(),
                ],
                ir.F32Type.get(),
            )

            @func.FuncOp.from_py_func(memref, memref, memref)
            def matmul(a, b, c):
                linalg.matmul(a, b, outs=[c])
                return

        module.operation.verify()

        pm = PassManager("builtin.module")
        pm.add("matmul-vectorization{vector-size=32}")
        pm.add("cse")
        pm.run(module.operation)

        # CHECK:      #map = affine_map<(d0) -> (d0)>
        # CHECK:      #map1 = affine_map<(d0) -> (d0 ceildiv 32)>
        # CHECK:      func.func @matmul(%[[A:.+]]: memref<?x?xf32>, %[[B:.+]]: memref<?x?xf32>, %[[C:.+]]: memref<?x?xf32>) {
        # CHECK:        %[[ZERO:.+]] = arith.constant 0 : index
        # CHECK:        %[[ONE:.+]] = arith.constant 1 : index
        # CHECK:        %[[C32:.+]] = arith.constant 32 : index
        # CHECK:        %[[CST:.+]] = arith.constant 0.000000e+00 : f32
        # CHECK:        %[[PASS_THRU:.+]] = vector.splat %[[CST]] : vector<32xf32>
        # CHECK:        %[[DIM:.+]] = memref.dim %[[A]], %[[ZERO]] : memref<?x?xf32>
        # CHECK:        %[[DIM_0:.+]] = memref.dim %[[B]], %[[ZERO]] : memref<?x?xf32>
        # CHECK:        %[[DIM_1:.+]] = memref.dim %[[B]], %[[ONE]] : memref<?x?xf32>
        # CHECK-NEXT:   affine.for %[[I:.+]] = #map(%[[ZERO]]) to #map(%[[DIM_0]]) {
        # CHECK-NEXT:     affine.for %[[J:.+]] = #map(%[[ZERO]]) to #map(%[[DIM]]) {
        # CHECK-NEXT:       affine.for %[[K:.+]] = #map(%[[ZERO]]) to #map1(%[[DIM_1]]) {
        # CHECK-NEXT:         %[[A_VAL:.+]] = memref.load %[[A]][%[[J]], %[[I]]] : memref<?x?xf32>
        # CHECK-NEXT:         %[[BROADCAST:.+]] = vector.broadcast %[[A_VAL]] : f32 to vector<32xf32>
        # CHECK-NEXT:         %[[CURRENT:.+]] = arith.muli %[[K]], %[[C32]] : index
        # CHECK-NEXT:         %[[REMAINDER:.+]] = arith.subi %[[DIM_1]], %[[CURRENT]] : index
        # CHECK-NEXT:         %[[COND:.+]] = arith.cmpi sge, %[[REMAINDER]], %[[C32]] : index
        # CHECK-NEXT:         scf.if %[[COND]] {
        # CHECK-NEXT:           %[[B_VEC:.+]] = affine.vector_load %[[B]][%[[I]], %[[K]] * 32] : memref<?x?xf32>, vector<32xf32>
        # CHECK-NEXT:           %[[C_VEC:.+]] = affine.vector_load %[[C]][%[[J]], %[[K]] * 32] : memref<?x?xf32>, vector<32xf32>
        # CHECK-NEXT:           %[[FMA:.+]] = vector.fma %[[BROADCAST]], %[[B_VEC]], %[[C_VEC]] : vector<32xf32>
        # CHECK-NEXT:           affine.vector_store %[[FMA]], %[[C]][%[[J]], %[[K]] * 32] : memref<?x?xf32>, vector<32xf32>
        # CHECK-NEXT:         } else {
        # CHECK-NEXT:           %[[MASK:.+]] = vector.create_mask %[[REMAINDER]] : vector<32xi1>
        # CHECK-NEXT:           %[[B_VEC:.+]] = vector.maskedload %[[B]][%[[I]], %[[CURRENT]]], %[[MASK]], %[[PASS_THRU]] : memref<?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        # CHECK-NEXT:           %[[C_VEC:.+]] = vector.maskedload %[[C]][%[[J]], %[[CURRENT]]], %[[MASK]], %[[PASS_THRU]] : memref<?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        # CHECK-NEXT:           %[[FMA:.+]] = vector.fma %[[BROADCAST]], %[[B_VEC]], %[[C_VEC]] : vector<32xf32>
        # CHECK-NEXT:           vector.maskedstore %[[C]][%[[J]], %[[CURRENT]]], %[[MASK]], %[[FMA]] : memref<?x?xf32>, vector<32xi1>, vector<32xf32>
        # CHECK-NEXT:         }
        # CHECK-NEXT:       }
        # CHECK-NEXT:     }
        # CHECK-NEXT:   }
        # CHECK-NEXT:   return
        # CHECK-NEXT: }

        print(module)
