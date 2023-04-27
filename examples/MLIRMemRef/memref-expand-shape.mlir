// RUN: buddy-opt %s \
// RUN:     --lower-affine -convert-memref-to-llvm \
// RUN:     --convert-arith-to-llvm  -convert-func-to-llvm \
// RUN:		   -reconcile-unrealized-casts \
// RUN:	| mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  memref.global "private" @gv : memref<4x4xf32> = dense<[[0., 1., 2., 3.],
                                                        [4., 5., 6., 7.],
                                                        [8., 9., 10., 12.],
                                                        [13., 14., 15., 16.]]>
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %mem0 = memref.get_global @gv : memref<4x4xf32>
    %mem1 = memref.cast %mem0 : memref<4x4xf32> to memref<?x?xf32>
    %mem2 = memref.expand_shape %mem1 [[0],[1,2,3]] : memref<?x?xf32> into memref<?x2x?x2xf32>
    %print_output0 = memref.cast %mem2 : memref<?x2x?x2xf32> to memref<*xf32>

    // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [4, 2, 1, 2] strides = [4, 2, 2, 1] data = 
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [
    // CHECK-SAME: [0,     1]
    // CHECK-SAME: ],
    // CHECK-NEXT: [
    // CHECK-SAME: [2,     3]
    // CHECK-SAME: ]
    // CHECK-SAME: ],
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SMAE: [4,     5]
    // CHECK-SAME: ],
    // CHECK-NEXT: [
    // CHECK-SAME: [6,     7]
    // CHECK-SAME: ]
    // CHECK-SAME: ],
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [8,     9]
    // CHECK-SAME: ],
    // CHECK-NEXT: [
    // CHECK-SAME: [10,     12]
    // CHECK-SAME: ]
    // CHECK-SAME: ],
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SMAE: [13,     14]
    // CHECK-SAME: ],
    // CHECK-NEXT: [
    // CHECK-SAME: [15,     16]
    // CHECK-SAME: ]
    // CHECK-SMAE: ]
    // CHECK-SAME: ]
    
    call @printMemrefF32(%print_output0) : (memref<*xf32>) -> ()
    %mem3 = memref.expand_shape %mem1 [[0,1],[2,3]] : memref<?x?xf32> into memref<?x2x?x2xf32>
    %print_output1 = memref.cast %mem3 : memref<?x2x?x2xf32> to memref<*xf32>

    // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [2, 2, 2, 2] strides = [8, 4, 2, 1] data = 
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SMAE: [
    // CHECK-SAME: [0,     1], 
    // CHECK-NEXT: [2,     3]
    // CHECK-SAME: ], 
    // CHECK-NEXT: [
    // CHECK-SAME: [4,     5],
    // CHECK-NEXT: [6,     7]
    // CHECK-SAME: ]
    // CHECK-SAME: ], 
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SMAE: [
    // CHECK-SAME: [8,     9], 
    // CHECK-NEXT: [10,     12]
    // CHECK-SAME: ], 
    // CHECK-NEXT: [
    // CHECK-SAME: [13,     14],
    // CHECK-NEXT: [15,     16]
    // CHECK-SAME: ]
    // CHECK-SAME: ]
    // CHECK-SAME: ]
    call @printMemrefF32(%print_output1) : (memref<*xf32>) -> ()
    %mem4 = memref.expand_shape %mem1 [[0,1],[2,3]] : memref<?x?xf32> into memref<1x?x?x2xf32>
    
    // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 4, 2, 2] strides = [16, 4, 2, 1] data = 
    // CHECK-NEXT: [
    // CHECK-SAME: [
    // CHECK-SAME: [
    // CHECK-SAME: [0,     1], 
    // CHECK-NEXT: [2,     3]
    // CHECK-SAME: ], 
    // CHECK-NEXT: [
    // CHECK-SAME: [4,     5], 
    // CHECK-NEXT: [6,     7]
    // CHECK-SAME: ],
    // CHECK-NEXT: [
    // CHECK-SAME: [8,     9], 
    // CHECK-NEXT: [10,     12]
    // CHECK-SAME: ], 
    // CHECK-NEXT: [
    // CHECK-SAME: [13,     14], 
    // CHECK-NEXT: [15,     16]
    // CHECK-SAME: ]
    // CHECK-SAME: ]
    // CHECK-SAME: ]
    %print_output2 = memref.cast %mem4 : memref<1x?x?x2xf32> to memref<*xf32>
    call @printMemrefF32(%print_output2) : (memref<*xf32>) -> ()
    return
  }
}
