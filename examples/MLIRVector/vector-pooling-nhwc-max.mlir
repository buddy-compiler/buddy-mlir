// RUN: buddy-opt %s \
// RUN:   -convert-vector-to-scf \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:		-convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN: 	-llvm-request-c-wrappers \
// RUN:		-convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 ceildiv 32)>
#map2 = affine_map<(d0, d1) -> (d0 + d1 - 1)>

module {
  memref.global "private" @kernel : memref<3x3xi32> = dense<0>

  func.func private @rtclock() -> f64
  func.func private @printMemrefI32(memref<*xi32>)
  func.func @pooling_nhwc_max(%arg0: memref<?x?x?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?x?x?xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = vector.splat %c0_i32 : vector<32xi32>
    %dim = memref.dim %arg1, %c0 : memref<?x?xi32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xi32>
    %dim_1 = memref.dim %arg2, %c0 : memref<?x?x?x?xi32>
    %dim_2 = memref.dim %arg2, %c1 : memref<?x?x?x?xi32>
    %dim_3 = memref.dim %arg2, %c2 : memref<?x?x?x?xi32>
    %dim_4 = memref.dim %arg2, %c3 : memref<?x?x?x?xi32>
    affine.for %arg3 = #map(%c0) to #map(%dim_1) {
      affine.for %arg4 = #map(%c0) to #map(%dim_4) {
        affine.for %arg5 = #map(%c0) to #map(%dim_2) {
          affine.for %arg6 = #map(%c0) to #map1(%dim_3) {
            %1 = arith.muli %arg6, %c32 : index
            %2 = arith.subi %dim_3, %1 : index
            %3 = arith.cmpi sge, %2, %c32 : index
            scf.if %3 {
              %4 = affine.vector_load %arg2[%arg3, %arg5, %arg6 * 32, %arg4] : memref<?x?x?x?xi32>, vector<32xi32>
              %5 = affine.for %arg7 = #map(%c0) to #map(%dim) iter_args(%arg8 = %4) -> (vector<32xi32>) {
                %6 = affine.for %arg9 = #map(%c0) to #map(%dim_0) iter_args(%arg10 = %arg8) -> (vector<32xi32>) {
                  %7 = affine.vector_load %arg0[%arg3, %arg7 + %arg5, %arg9 + %arg6 * 32, %arg4] : memref<?x?x?x?xi32>, vector<32xi32>
                  %8 = arith.maxsi %7, %arg10 : vector<32xi32>
                  affine.yield %8 : vector<32xi32>
                }
                affine.yield %6 : vector<32xi32>
              }
              affine.vector_store %5, %arg2[%arg3, %arg5, %arg6 * 32, %arg4] : memref<?x?x?x?xi32>, vector<32xi32>
            } else {
              %4 = vector.create_mask %2 : vector<32xi1>
              %5 = vector.maskedload %arg2[%arg3, %arg5, %1, %arg4], %4, %0 : memref<?x?x?x?xi32>, vector<32xi1>, vector<32xi32> into vector<32xi32>
              %6 = affine.for %arg7 = #map(%c0) to #map(%dim) iter_args(%arg8 = %5) -> (vector<32xi32>) {
                %7 = affine.for %arg9 = #map(%c0) to #map(%dim_0) iter_args(%arg10 = %arg8) -> (vector<32xi32>) {
                  %8 = arith.addi %arg5, %arg7 : index
                  %9 = arith.addi %arg9, %1 : index
                  %10 = vector.maskedload %arg0[%arg3, %8, %9, %arg4], %4, %0 : memref<?x?x?x?xi32>, vector<32xi1>, vector<32xi32> into vector<32xi32>
                  %11 = arith.maxsi %10, %arg10 : vector<32xi32>
                  affine.yield %11 : vector<32xi32>
                }
                affine.yield %7 : vector<32xi32>
              }
              vector.maskedstore %arg2[%arg3, %arg5, %1, %arg4], %4, %6 : memref<?x?x?x?xi32>, vector<32xi1>, vector<32xi32>
            }
          }
        }
      }
    }
    return
  }

  func.func @alloc_i32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: i32) -> memref<?x?x?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xi32>
    scf.for %arg5 = %c0 to %arg0 step %c1 {
      scf.for %arg6 = %c0 to %arg1 step %c1 {
        scf.for %arg7 = %c0 to %arg2 step %c1 {
          scf.for %arg8 = %c0 to %arg3 step %c1 {
            memref.store %arg4, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xi32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xi32>
  }

  func.func @main() {
    %N = arith.constant 1 : index
    %current_v1 = arith.constant 3 : index
    %current_v2 = arith.constant 126 : index
    %current_v0 = affine.apply #map2(%current_v2, %current_v1)
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %kernel = memref.get_global @kernel : memref<3x3xi32>

    %a = call @alloc_i32(%N, %current_v0, %current_v0, %N, %c1) : (index, index, index, index, i32) -> memref<?x?x?x?xi32>
    %b = memref.cast %kernel : memref<3x3xi32> to memref<?x?xi32>
    %c = call @alloc_i32(%N, %current_v2, %current_v2, %N, %c0) : (index, index, index, index, i32) -> memref<?x?x?x?xi32>

    %t0 = call @rtclock() : () -> f64
    call @pooling_nhwc_max(%a, %b, %c) : (memref<?x?x?x?xi32>, memref<?x?xi32>, memref<?x?x?x?xi32>) -> ()
    %t1 = call @rtclock() : () -> f64
    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref
    // CHECK: [
    // CHECK: [
    // CHECK: [
    // CHECK: [1],
    %print_c = memref.cast %c : memref<?x?x?x?xi32> to memref<*xi32>
    call @printMemrefI32(%print_c) : (memref<*xi32>) -> ()

    %time = arith.subf %t1, %t0 : f64
    vector.print %time : f64

    memref.dealloc %a : memref<?x?x?x?xi32>
    memref.dealloc %c : memref<?x?x?x?xi32>

    return 
  }
}
