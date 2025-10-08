# RUN: %PYTHON %s | FileCheck %s

from buddy_mlir.dialects import arith, func, vir, memref
from buddy_mlir import ir
from buddy_mlir.passmanager import PassManager


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testVIROperations
@run
def testVIROperations():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(
            """
            memref.global "private" @gv : memref<10xf32> = dense<[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.]>
            func.func private @printMemrefF32(memref<*xf32>)

            func.func @main() {
                %vl = arith.constant 5 : index
                %f1 = arith.constant 1.0 : f32
                %mem = memref.get_global @gv : memref<10xf32>
                %c0 = arith.constant 0 : index
                %c5 = arith.constant 5 : index

                vir.set_vl %vl : index {
                    %v1 = vir.constant { value = 2.0 : f32 } : !vir.vec<?xf32>
                    %v2 = vir.broadcast %f1 : f32 -> !vir.vec<?xf32>
                    vir.store %v1, %mem[%c0] : !vir.vec<?xf32> -> memref<10xf32>
                    vir.store %v2, %mem[%c5] : !vir.vec<?xf32> -> memref<10xf32>
                    vector.yield
                }

                %print_mem =  memref.cast %mem : memref<10xf32> to memref<*xf32>
                call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()

                return
            }
            """
        )

        module.operation.verify()

        pm = PassManager("builtin.module")
        pm.add("lower-vir-to-vector")
        pm.run(module.operation)
        
        # CHECK: #map = affine_map<(d0) -> (d0)>
        # CHECK: func.func @main() {
        # CHECK:   %[[VL:.*]] = arith.constant 5 : index
        # CHECK:   %[[F1:.*]] = arith.constant 1.000000e+00 : f32
        # CHECK:   %[[MEM:.*]] = memref.get_global @gv : memref<10xf32>
        # CHECK:   %[[C0:.*]] = arith.constant 0 : index
        # CHECK:   %[[C5:.*]] = arith.constant 5 : index
        # CHECK:   %[[C256:.*]] = arith.constant 256 : index
        # CHECK:   affine.for %{{.*}} = #map(%{{.*}}) to #map(%{{.*}}) step 256 {
        # CHECK:     %[[CONST_VEC:.*]] = arith.constant dense<2.000000e+00> : vector<256xf32>
        # CHECK:     %[[BROADCAST_VEC:.*]] = vector.broadcast %[[F1]] : f32 to vector<256xf32>
        # CHECK:     vector.store %[[CONST_VEC]], %[[MEM]][%{{.*}}] : memref<10xf32>, vector<256xf32>
        # CHECK:     vector.store %[[BROADCAST_VEC]], %[[MEM]][%{{.*}}] : memref<10xf32>, vector<256xf32>
        # CHECK:   }
        # CHECK:   affine.for %{{.*}} = #map(%{{.*}}) to #map(%[[VL]]) {
        # CHECK:     %[[SCALAR_CONST:.*]] = arith.constant 2.000000e+00 : f32
        # CHECK:     memref.store %[[SCALAR_CONST]], %[[MEM]][%{{.*}}] : memref<10xf32>
        # CHECK:     memref.store %[[F1]], %[[MEM]][%{{.*}}] : memref<10xf32>
        # CHECK:   }
        print(module)
