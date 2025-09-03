// RUN: buddy-opt %s \
// RUN:     -lower-affine -convert-vector-to-scf -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm \
// RUN:		  -finalize-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm \
// RUN:		  -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  memref.global "private" @gv : memref<5x5xf32> = dense<[[0. , 1. , 2. , 3. , 4.],
                                                        [10., 11., 12., 13. , 14.],
                                                        [20., 21., 22., 23. , 24.],
                                                        [30., 31., 32., 33. , 34.],
                                                        [40., 41., 42., 43. , 44.]]>

  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %i0 = arith.constant 1 : index
    %j0 = arith.constant 1 : index
    %i1 = arith.constant 0 : index
    %j1 = arith.constant 0 : index
    %mem = memref.get_global @gv : memref<5x5xf32>
    // Load.
    // Method one.
    %v0 = affine.vector_load %mem[%i0, %j0] : memref<5x5xf32>, vector<2xf32>
    // CHECK: ( 11, 12 )
    vector.print %v0 : vector<2xf32>
    // Method two.
    %v1 = vector.load %mem[%i0, %j0] : memref<5x5xf32>, vector<4xf32>
    // CHECK: ( 11, 12, 13, 14 )
    vector.print %v1 : vector<4xf32>
    // Store.
    // Method one.
    affine.vector_store %v0, %mem[%i1, %j1] : memref<5x5xf32>, vector<2xf32>
    %print_out1 = memref.cast %mem : memref<5x5xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [5, 5] strides = [5, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME:  [11, 12, 2, 3, 4],
    // CHECK-NEXT:  [10, 11, 12, 13, 14],
    // CHECK-NEXT:  [20, 21, 22, 23, 24],
    // CHECK-NEXT:  [30, 31, 32, 33, 34],
    // CHECK-NEXT:  [40, 41, 42, 43, 44]
    // CHECK-SAME: ]
    func.call @printMemrefF32(%print_out1) : (memref<*xf32>) -> ()
    // Method two.
    affine.vector_store %v1, %mem[%i1, %j1] : memref<5x5xf32>, vector<4xf32>
    %print_out2 = memref.cast %mem : memref<5x5xf32> to memref<*xf32>
    // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [5, 5] strides = [5, 1] data =
    // CHECK-NEXT: [
    // CHECK-SAME:  [11, 12, 13, 14, 4],
    // CHECK-NEXT:  [10, 11, 12, 13, 14],
    // CHECK-NEXT:  [20, 21, 22, 23, 24],
    // CHECK-NEXT:  [30, 31, 32, 33, 34],
    // CHECK-NEXT:  [40, 41, 42, 43, 44]
    // CHECK-SAME: ]
    func.call @printMemrefF32(%print_out2) : (memref<*xf32>) -> ()
    func.return
    }
}
