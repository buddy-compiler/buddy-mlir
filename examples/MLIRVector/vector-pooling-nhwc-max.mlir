// RUN: buddy-opt %s \
// RUN:     -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:		-convert-vector-to-llvm -finalize-memref-to-llvm \
// RUN:		-convert-func-to-llvm -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 ceildiv 32)>

module{
    memref.global "private" @input : memref<1x4x4x1xf32> = 
        dense<[[[[1.], [2.], [3.], [4.]], 
                [[4.], [3.], [5.], [8.]],
                [[4.], [5.], [3.], [8.]], 
                [[9.], [8.], [5.], [1.]]]]>
    memref.global "private" @kernel : memref<2x2xf32> = dense<0.0>
    memref.global "private" @output : memref<1x3x3x1xf32> = dense<0.0>

    func.func private @printMemrefF32(memref<*xf32>)

    func.func @pooling_nhwc_max(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = vector.splat %cst : vector<32xf32>
      %dim = memref.dim %arg1, %c0 : memref<?x?xf32>
      %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x?x?x?xf32>
      %dim_2 = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
      %dim_3 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
      %dim_4 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32>
      affine.for %tmp0 = #map(%c0) to #map(%dim_1) {
        affine.for %tmp1 = #map(%c0) to #map(%dim_4) {
          affine.for %tmp2 = #map(%c0) to #map(%dim_2) {
            affine.for %tmp3 = #map(%c0) to #map(%dim) {
              affine.for %tmp4 = #map(%c0) to #map(%dim_0) {
                affine.for %tmp5 = #map(%c0) to #map1(%dim_3) { 
                  %1 = arith.muli %tmp5, %c32 : index
                  %2 = arith.subi %dim_3, %1 : index
                  %3 = arith.cmpi sge, %2, %c32 : index
                  scf.if %3 {
                    %4 = affine.vector_load %arg0[%tmp0, %tmp2 + %tmp3, %tmp4 + %tmp5 * 32, %tmp1] : memref<?x?x?x?xf32>, vector<32xf32>
                    %5 = affine.vector_load %arg2[%tmp0,  %tmp2, %tmp5 * 32, %tmp1] : memref<?x?x?x?xf32>, vector<32xf32> 
                    %6 = arith.maximumf %4, %5 : vector<32xf32> 
                    affine.vector_store %6, %arg2[%tmp0, %tmp2, %tmp5 * 32, %tmp1] : memref<?x?x?x?xf32>, vector<32xf32>
                  } else {
                    %7 = vector.create_mask %2 : vector<32xi1>
                    %8 = arith.addi %tmp2, %tmp3 : index
                    %9 = arith.muli %tmp5, %c32 : index
                    %10 = arith.addi %tmp4, %9 : index
                    %11 = vector.maskedload %arg0[%tmp0, %8, %10, %tmp1], %7, %0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
                    %12 = vector.maskedload %arg2[%tmp0, %tmp2, %9, %tmp1], %7, %0 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>   
                    %13 = arith.maximumf %11, %12 : vector<32xf32>
                    vector.maskedstore %arg2[%tmp0, %tmp2, %9, %tmp1], %7, %13 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
                  }
                }
              }
            }
          }
        }
      }
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
      // CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
      // CHECK-NEXT: [[
      // CHECK-SAME:  [[4], [5], [8]],
      // CHECK-NEXT:  [[5], [5], [8]],
      // CHECK-NEXT:  [[9], [8], [8]]
      // CHECK-SAME: ]]
      %print_c = memref.cast %c : memref<?x?x?x?xf32> to memref<*xf32>
      call @printMemrefF32(%print_c) : (memref<*xf32>) -> ()

      return 
    }
}
