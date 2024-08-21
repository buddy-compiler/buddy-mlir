// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-loops -lower-affine -convert-scf-to-cf \
// RUN:		-convert-vector-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm \
// RUN:		-convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module{
    memref.global "private" @input : memref<1x4x4x1xf32> = 
        dense<[[[[1.], [2.], [3.], [4.]], 
                [[4.], [3.], [5.], [8.]],
                [[4.], [5.], [3.], [8.]], 
                [[9.], [8.], [5.], [1.]]]]>
    memref.global "private" @kernel : memref<2x2xf32> = dense<0.0>
    memref.global "private" @output : memref<1x3x3x1xf32> = dense<0.0>

    func.func private @printMemrefF32(memref<*xf32>)

    func.func @pooling_nhwc_max(%a : memref<?x?x?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?x?x?xf32>) {
      linalg.pooling_nhwc_max  
        ins(%a, %b : memref<?x?x?x?xf32>, memref<?x?xf32>) 
        outs(%c : memref<?x?x?x?xf32>)
      return
    }

    func.func @main(){
      %input = memref.get_global @input : memref<1x4x4x1xf32>
      %kernel = memref.get_global @kernel : memref<2x2xf32>
      %output = memref.get_global @output : memref<1x3x3x1xf32>

      %a = memref.cast %input : memref<1x4x4x1xf32> to memref<?x?x?x?xf32>
      %b = memref.cast %kernel : memref<2x2xf32> to memref<?x?xf32>
      %c = memref.cast %output : memref<1x3x3x1xf32> to memref<?x?x?x?xf32>

      call @pooling_nhwc_max(%a, %b, %c) : (memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>) -> ()
      // Print output.
      // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 3, 3, 1] strides = [9, 3, 1, 1] data =
      // CHECK-NEXT: [[[[4],
      // CHECK-SAME:    [5], 
      // CHECK-SAME:    [8]],
      // CHECK-NEXT:   [[5],
      // CHECK-NEXT:    [5],
      // CHECK-NEXT:    [8]],
      // CHECK-NEXT:   [[9],
      // CHECK-NEXT:    [8],
      // CHECK-NEXT:    [8]]]]
      %print_c = memref.cast %c : memref<?x?x?x?xf32> to memref<*xf32>
      call @printMemrefF32(%print_c) : (memref<*xf32>) -> ()

      return 
    }
}
