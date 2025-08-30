# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir.dialects import func, linalg
from buddy_mlir import ir
from buddy_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testConv2DVectorize
@run
def testConv2DVectorize():
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
            def conv2d(input, kernel, output):
                linalg.conv_2d(input, kernel, outs=[output])
                return

        module.operation.verify()

        pm = PassManager("builtin.module")
        pm.add("conv-vectorization{strip-mining=32}")
        pm.add("cse")
        pm.run(module.operation)

        # Check the affine maps and function signature
        # CHECK:      #map = affine_map<(d0) -> (d0)>
        # CHECK:      #map1 = affine_map<(d0) -> (d0 ceildiv 32)>
        # CHECK:      func.func @conv2d(%[[IN:.+]]: memref<?x?xf32>, %[[KERNEL:.+]]: memref<?x?xf32>, %[[OUT:.+]]: memref<?x?xf32>) {
        # CHECK:        %[[ZERO:.+]] = arith.constant 0 : index
        # CHECK:        %[[ONE:.+]] = arith.constant 1 : index
        # CHECK:        %[[C32:.+]] = arith.constant 32 : index
        # CHECK:        %[[CST:.+]] = arith.constant 0.000000e+00 : f32

        # Check the pass-through vector for masked loads and stores
        # CHECK:        %[[PASS_THRU:.+]] = vector.splat %[[CST]] : vector<32xf32>

        # Check the dims
        # CHECK:        %[[KERNEL_DIM0:.+]] = memref.dim %[[KERNEL]], %[[ZERO]] : memref<?x?xf32>
        # CHECK:        %[[KERNEL_DIM1:.+]] = memref.dim %[[KERNEL]], %[[ONE]] : memref<?x?xf32>
        # CHECK:        %[[OUT_DIM0:.+]] = memref.dim %[[OUT]], %[[ZERO]] : memref<?x?xf32>
        # CHECK:        %[[OUT_DIM1:.+]] = memref.dim %[[OUT]], %[[ONE]] : memref<?x?xf32>

        # Check the vectorized loop nest
        # CHECK:        affine.for %[[I:.+]] = #map(%[[ZERO]]) to #map(%[[OUT_DIM0]]) {
        # CHECK-NEXT:     affine.for %[[J:.+]] = #map(%[[ZERO]]) to #map(%[[KERNEL_DIM0]]) {
        # CHECK-NEXT:       affine.for %[[K:.+]] = #map(%[[ZERO]]) to #map(%[[KERNEL_DIM1]]) {
        # CHECK-NEXT:        affine.for %[[L:.+]] = #map(%[[ZERO]]) to #map1(%[[OUT_DIM1]]) {
        # CHECK:               %[[KERNEL_VAL:.+]] = memref.load %[[KERNEL]][%[[J]], %[[K]]] : memref<?x?xf32>
        # CHECK:               %[[COND:.+]] = arith.cmpf one, %[[KERNEL_VAL]], %{{.+}} : f32
        # CHECK:               scf.if %[[COND]] {
        # CHECK-NEXT:            %[[BROADCAST:.+]] = vector.broadcast %[[KERNEL_VAL]] : f32 to vector<32xf32>
        # CHECK-NEXT:            %[[CURRENT:.+]] = arith.muli %[[L]], %[[C32]] : index
        # CHECK-NEXT:            %[[REMAINDER:.+]] = arith.subi %[[OUT_DIM1]], %[[CURRENT]] : index
        # CHECK-NEXT:            %[[COND:.+]] = arith.cmpi sge, %[[REMAINDER]], %[[C32]] : index
        # CHECK-NEXT:            scf.if %[[COND]] {
        # CHECK-NEXT:              %[[IN_VEC:.+]] = affine.vector_load %[[IN]][%[[I]] + %[[J]], %[[K]] + %[[L]] * 32] : memref<?x?xf32>, vector<32xf32>
        # CHECK-NEXT:              %[[OUT_VEC:.+]] = affine.vector_load %[[OUT]][%[[I]], %[[L]] * 32] : memref<?x?xf32>, vector<32xf32>
        # CHECK-NEXT:              %[[FMA:.+]] = vector.fma %[[IN_VEC]], %[[BROADCAST]], %[[OUT_VEC]] : vector<32xf32>
        # CHECK-NEXT:              affine.vector_store %[[FMA]], %[[OUT]][%[[I]], %[[L]] * 32] : memref<?x?xf32>, vector<32xf32>
        # CHECK-NEXT:            } else {
        # CHECK-NEXT:              %[[MASK:.+]] = vector.create_mask %[[REMAINDER]] : vector<32xi1>
        # CHECK-NEXT:              %[[INPUT_ROW:.+]] = arith.addi %[[I]], %[[J]] : index
        # CHECK-NEXT:              %[[INPUT_COL:.+]] = arith.addi %[[K]], %[[CURRENT]] : index
        # CHECK-NEXT:              %[[IN_VEC:.+]] = vector.maskedload %[[IN]][%[[INPUT_ROW]], %[[INPUT_COL]]], %[[MASK]], %[[PASS_THRU]] : memref<?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        # CHECK-NEXT:              %[[OUT_VEC:.+]] = vector.maskedload %[[OUT]][%[[I]], %[[CURRENT]]], %[[MASK]], %[[PASS_THRU]] :  memref<?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        # CHECK-NEXT:              %[[FMA:.+]] = vector.fma %[[IN_VEC]], %[[BROADCAST]], %[[OUT_VEC]] : vector<32xf32>
        # CHECK-NEXT:              vector.maskedstore %[[OUT]][%[[I]], %[[CURRENT]]], %[[MASK]], %[[FMA]] : memref<?x?xf32>, vector<32xi1>, vector<32xf32>
        # CHECK-NEXT:            }
        # CHECK-NEXT:          }
        # CHECK-NEXT:        }
        # CHECK-NEXT:       }
        # CHECK-NEXT:     }
        # CHECK-NEXT:   }
        # CHECK-NEXT:   return
        # CHECK-NEXT: }
        print(module)
