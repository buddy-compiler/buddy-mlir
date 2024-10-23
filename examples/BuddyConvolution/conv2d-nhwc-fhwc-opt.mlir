// RUN: buddy-opt %s \
// RUN:   -convert-vector-to-scf \
// RUN:   -lower-affine \
// RUN:   -arith-bufferize \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-vector-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// Using `8` as the vector size.
#map = affine_map<(d0) -> (d0 ceildiv 16)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d2)>
#map1 = affine_map<(d0, d1) -> (d0 + d1)>
#map2 = affine_map<(d0, d1) -> (d0 + d1 * 16)>
#map3 = affine_map<(d0) -> (d0 * 16)>

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64

  func.func @conv_2d_nhwc_fhwc(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 16 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %f0 : vector<16xf32>
    %n = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %h_i = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %w_i = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
    %c = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
    %f = memref.dim %arg1, %c0 : memref<?x?x?x?xf32>
    %h_k = memref.dim %arg1, %c1 : memref<?x?x?x?xf32>
    %w_k = memref.dim %arg1, %c2 : memref<?x?x?x?xf32>
    %h_o = memref.dim %arg2, %c1 : memref<?x?x?x?xf32>
    %w_o = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>

    // Output is NHoWoF
    affine.for %idx_n = %c0 to %n {
      affine.for %idx_f = %c0 to %f {
        affine.for %idx_c = %c0 to %c {
          affine.for %idx_h_o = %c0 to %h_o {
            affine.for %idx_w_o = %c0 to #map(%w_o) {         
              %1 = arith.muli %idx_w_o, %c32 : index
              %2 = arith.subi %w_o, %1 : index
              %3 = arith.cmpi sge, %2, %c32 : index
              scf.if %3 {
                // %arg2[%n, %h_o, %w_o*16, %f]
                %output_vec = vector.transfer_read %arg2[%idx_n, %idx_h_o, %1, %idx_f], %f0
                      { permutation_map = #map0 } : memref<?x?x?x?xf32>, vector<16xf32>
                %5 = affine.for %idx_h_k = %c0 to %h_k iter_args(%arg8 = %output_vec) -> (vector<16xf32>) {        // %h_k
                  %6 = affine.for %idx_w_k = %c0 to %w_k iter_args(%arg10 = %arg8) -> (vector<16xf32>) {           // %w_k
                    // %arg1[%f, %h_k, %w_k, %c]
                    %kernel_ele = memref.load %arg1[%idx_f, %idx_h_k, %idx_w_k, %idx_c] : memref<?x?x?x?xf32>
                    %kernel_vec = vector.broadcast %kernel_ele : f32 to vector<16xf32>
                    %in_iter_h = affine.apply #map1 (%idx_h_k, %idx_h_o)
                    %in_iter_w = affine.apply #map2 (%idx_w_k, %idx_w_o)
                    // %arg0[%n, %h_k+%h_o, %w_k+%w_o*16, %c]
                    %input_vec = vector.transfer_read %arg0[%idx_n, %in_iter_h, %in_iter_w, %idx_c], %f0
                      { permutation_map = #map0 } : memref<?x?x?x?xf32>, vector<16xf32>
                    %res_vec = vector.fma %kernel_vec, %input_vec, %arg10 : vector<16xf32>
                    affine.yield %res_vec : vector<16xf32>
                  }
                  affine.yield %6 : vector<16xf32>
                }
                 vector.transfer_write %5, %arg2[%idx_n, %idx_h_o, %1, %idx_f]
                  { permutation_map = #map0 } : vector<16xf32>, memref<?x?x?x?xf32>
              } else {
                %9 = vector.create_mask %2 : vector<16xi1>
                // %arg2[%n, %h_o, %w_o*16, %f]
                %output_vec = vector.transfer_read %arg2[%idx_n, %idx_h_o, %1, %idx_f], %f0, %9
                      { permutation_map = #map0 } : memref<?x?x?x?xf32>, vector<16xf32>
                %5 = affine.for %idx_h_k = %c0 to %h_k iter_args(%arg8 = %output_vec) -> (vector<16xf32>) {        // %h_k
                  %6 = affine.for %idx_w_k = %c0 to %w_k iter_args(%arg10 = %arg8) -> (vector<16xf32>) {           // %w_k
                    // %arg1[%f, %h_k, %w_k, %c]
                    %kernel_ele = memref.load %arg1[%idx_f, %idx_h_k, %idx_w_k, %idx_c] : memref<?x?x?x?xf32>
                    %kernel_vec = vector.broadcast %kernel_ele : f32 to vector<16xf32>
                    %in_iter_h = affine.apply #map1 (%idx_h_k, %idx_h_o)
                    %in_iter_w = affine.apply #map2 (%idx_w_k, %idx_w_o)
                    // %arg0[%n, %h_k+%h_o, %w_k+%w_o*16, %c]
                    %input_vec = vector.transfer_read %arg0[%idx_n, %in_iter_h, %in_iter_w, %idx_c], %f0, %9
                      { permutation_map = #map0 } : memref<?x?x?x?xf32>, vector<16xf32>
                    %res_vec = vector.fma %kernel_vec, %input_vec, %arg10 : vector<16xf32>
                    affine.yield %res_vec : vector<16xf32>
                  }
                  affine.yield %6 : vector<16xf32>
                }
                vector.transfer_write %5, %arg2[%idx_n, %idx_h_o, %1, %idx_f], %9
                { permutation_map = #map0 } : vector<16xf32>, memref<?x?x?x?xf32>
              }
            }
          }
        }
      }
    }

    return
  }

  func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    scf.for %idx0 = %c0 to %arg0 step %c1 {
      scf.for %idx1 = %c0 to %arg1 step %c1 {
        scf.for %idx2 = %c0 to %arg2 step %c1 {
          scf.for %idx3 = %c0 to %arg3 step %c1 {
            memref.store %arg4, %0[%idx0, %idx1, %idx2, %idx3] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }

  func.func @main() {
    %f0 = arith.constant 0.000000e+00 : f32
    %f2 = arith.constant 2.000000e+00 : f32
    %f3 = arith.constant 3.000000e+00 : f32

    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c8 = arith.constant 8 : index
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    %c28 = arith.constant 28 : index
    
    %v0 = call @alloc_f32(%c1, %c28, %c28, %c5, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %v1 = call @alloc_f32(%c6, %c5, %c5, %c5, %f3) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %v2 = call @alloc_f32(%c1, %c24, %c24, %c6, %f0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    %t_start = call @rtclock() : () -> f64
    call @conv_2d_nhwc_fhwc(%v0, %v1, %v2) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
    %t_end = call @rtclock() : () -> f64

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref
    // CHECK: [
    // CHECK: [
    // CHECK: [
    // CHECK: [750{{(, 750)*}}],
    %print_v2 = memref.cast %v2 : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_v2) : (memref<*xf32>) -> ()

    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64

    memref.dealloc %v0 : memref<?x?x?x?xf32>
    memref.dealloc %v1 : memref<?x?x?x?xf32>
    memref.dealloc %v2 : memref<?x?x?x?xf32>

    return
  }
}
