// RUN: buddy-opt %s -lower-bud \
// RUN: | FileCheck %s

module {
  memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                         [10., 11., 12., 13.],
                                                         [20., 21., 22., 23.],
                                                         [30., 31., 32., 33.]]>
  %mem = memref.get_global @gv : memref<4x4xf32>
  // CHECK: %[[CONSTANT_0:.*]] = arith.constant 0 : index
  // CHECK: %[[CONSTANT_1:.*]] = arith.constant 1 : index
  // CHECK: %{{.*}} = memref.load %{{.*}}[%[[CONSTANT_0]], %[[CONSTANT_1]]] : memref<4x4xf32>
  %res = bud.test_array_attr %mem {coordinate = [0, 1]} : memref<4x4xf32>, f32
}
