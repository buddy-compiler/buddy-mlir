// RUN: buddy-opt %s \
// RUN:     -lower-affine -finalize-memref-to-llvm \
// RUN:     -convert-scf-to-cf -convert-cf-to-llvm -convert-func-to-llvm  \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %c0 = arith.constant 2 : index
    %c1 = arith.constant 0 : index
    %c2 = arith.constant 8 : index
    %c3 = arith.constant 1 : index
    %f0 = arith.constant 520. : f32
    %f1 = arith.constant 1314. : f32
    %f2 = arith.constant 0. : f32
    %mem0 = memref.alloc() : memref<8xf32>
    scf.for %i = %c1 to %c2 step %c3 {
      memref.store %f2, %mem0[%i] : memref<8xf32>
    }
    %print_out0 = memref.cast %mem0 : memref<8xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [8] strides = [1] data =
    // CHECK-NEXT: [0,  0,  0,  0,  0,  0,  0,  0]
    call  @printMemrefF32(%print_out0) : (memref<*xf32>) -> ()
    %mem1 = memref.reinterpret_cast %mem0 to
    offset : [1],
    sizes  : [4, 2],
    strides : [1, 1]
    : memref<8xf32> to memref<4x2xf32, strided<[1, 1], offset: 1>>
    %print_out1 = memref.cast %mem1 : memref<4x2xf32, strided<[1, 1], offset: 1>> to memref<*xf32>
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 1 sizes = [4, 2] strides = [1, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [0,   0],
    // CHECK-NEXT: [0,   0],
    // CHECK-NEXT: [0,   0],
    // CHECK-NEXT: [0,   0]
    // CHECK-SAME: ]
    call @printMemrefF32(%print_out1) : (memref<*xf32>) -> ()
    %mem2 = memref.reinterpret_cast %mem1 to
    offset : [0],
    sizes : [%c0, 4],
    strides : [1, %c0]
    : memref<4x2xf32, strided<[1, 1], offset: 1>> to memref<?x4xf32, strided<[1, ?], offset: 0>>
    %print_out2 = memref.cast %mem2 : memref<?x4xf32, strided<[1, ?], offset: 0>> to memref<*xf32>
    // CHECK:  Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [2, 4] strides = [1, 2] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [0,   0,   0,   0],
    // CHECK-NEXT:  [0,   0,   0,   0]
    // CHECK-SAME: ]
    call @printMemrefF32(%print_out2) : (memref<*xf32>) -> ()
    affine.store %f1, %mem0[%c0] :  memref<8xf32>
    affine.store %f0, %mem0[%c1] :  memref<8xf32>
    %print_out3 =  memref.cast %mem0 : memref<8xf32> to memref<*xf32>
    %print_out4 =  memref.cast %mem1 : memref<4x2xf32, strided<[1, 1], offset: 1>> to memref<*xf32>
    %print_out5 =  memref.cast %mem2 : memref<?x4xf32, strided<[1, ?], offset: 0>> to memref<*xf32>
    // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [8] strides = [1] data =
    // CHECK-NEXT: [520,  0,  1314,  0,  0,  0,  0,  0]
    call @printMemrefF32(%print_out3) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 1 sizes = [4, 2] strides = [1, 1] data =
    // CHECK-NEXT:[
    // CHECK-SAME: [0,   1314],
    // CHECK-NEXT: [1314,   0],
    // CHECK-NEXT: [0,   0],
    // CHECK-NEXT: [0,   0]
    // CHECK-SAME: ]
    call @printMemrefF32(%print_out4) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [2, 4] strides = [1, 2] data =
    // CHECK-NEXT: [
    // CHECK-SAME: [520,   1314,   0,   0],
    // CHECK-NEXT: [0,   0,   0,   0]
    // CHECK-SAME: ]
    call @printMemrefF32(%print_out5) : (memref<*xf32>) -> ()
    memref.dealloc  %mem0 : memref<8xf32>
    return
  }
}
