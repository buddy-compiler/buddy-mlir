// RUN: buddy-opt  %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -cse \
// RUN:   -lower-affine \
// RUN:   -convert-vector-to-scf \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -expand-strided-metadata \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

// HACK: max(NC, MC, KC) must be greater than min(MR, NR) to prevent out-of-bounds access.
#cond_set_a = affine_set<(i, j)[M, K] : (M - 1 - i >= 0, K - 1 - j >= 0)>
#cond_set_b = affine_set<(i, j)[N, K] : (K - 1 - i >= 0, N - 1 - j >= 0)>
#micro_kernel_offset = affine_map<(off, idx, p)[KC]->(off + idx*KC + p)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64
  
  func.func @pack_a(%a : memref<?x?xf32>, %a_packed : memref<?xf32>, %i_c: index, %p_c: index) {
    %cf0 = arith.constant 0. : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %KC = arith.constant 128 : index
    %MC = arith.constant 256 : index

    %M = memref.dim %a, %c0 : memref<?x?xf32>
    %K = memref.dim %a, %c1 : memref<?x?xf32>

    affine.for %j = 0 to %KC { // KC
      affine.for %i = 0 to %MC { // MC
        %i_global = affine.apply affine_map<(i_c,i)->(i_c+i)>(%i_c, %i)
        %j_global = affine.apply affine_map<(p_c,j)->(p_c+j)>(%p_c, %j)

        %idx = affine.apply affine_map<(i,j)[KC]->(i*KC + j)>(%i, %j)[%KC]

        affine.if #cond_set_a(%i_global, %j_global)[%M, %K] {
          %val = affine.load %a[%i_global, %j_global] : memref<?x?xf32>
          affine.store %val, %a_packed[%idx] : memref<?xf32>
        } else {
          affine.store %cf0, %a_packed[%idx] : memref<?xf32>
        }
      }
    }
    return
  }

  func.func @pack_b(%b : memref<?x?xf32>, %b_packed : memref<?xf32>, %j_c: index, %p_c: index) {
    %cf0 = arith.constant 0. : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %KC = arith.constant 128 : index
    %NC = arith.constant 1024 : index

    %K = memref.dim %b, %c0 : memref<?x?xf32>
    %N = memref.dim %b, %c1 : memref<?x?xf32>

    affine.for %i = 0 to %KC { // KC
      affine.for %j = 0 to %NC { // MC
        %i_global = affine.apply affine_map<(p_c,i)->(p_c+i)>(%p_c, %i)
        %j_global = affine.apply affine_map<(j_c,j)->(j_c+j)>(%j_c, %j)

        %idx = affine.apply affine_map<(i,j)[KC]->(i + j*KC)>(%i, %j)[%KC]

        affine.if #cond_set_b(%i_global, %j_global)[%N, %K] {
          %val = affine.load %b[%i_global, %j_global] : memref<?x?xf32>
          affine.store %val, %b_packed[%idx] : memref<?xf32>
        } else {
          affine.store %cf0, %b_packed[%idx] : memref<?xf32>
        }
      }
    }
    return
  }

  func.func @micro_kernel(%a_packed : memref<?xf32>, %b_packed : memref<?xf32>, %c_sub : memref<?xf32>, %a_offset: index, %b_offset: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index

  %KC = arith.constant 128 : index
  %NR = arith.constant 8   : index

  %f0 = arith.constant 0.0 : f32
  %z  = vector.broadcast %f0 : f32 to vector<8xf32>

  %acc0_fin, %acc1_fin, %acc2_fin, %acc3_fin =
    scf.for %p = %c0 to %KC step %c1
      iter_args(%acc0 = %z, %acc1 = %z, %acc2 = %z, %acc3 = %z)
      -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {

        %binit = arith.constant dense<0.0> : vector<8xf32>

        %bidx0 = affine.apply #micro_kernel_offset(%b_offset, %c0, %p)[%KC]
        %b0 = memref.load %b_packed[%bidx0] : memref<?xf32>
        %bv0 = vector.insert %b0, %binit[0] : f32 into vector<8xf32>

        %bidx1 = affine.apply #micro_kernel_offset(%b_offset, %c1, %p)[%KC]
        %b1 = memref.load %b_packed[%bidx1] : memref<?xf32>
        %bv1 = vector.insert %b1, %bv0[1] : f32 into vector<8xf32>

        %bidx2 = affine.apply #micro_kernel_offset(%b_offset, %c2, %p)[%KC]
        %b2 = memref.load %b_packed[%bidx2] : memref<?xf32>
        %bv2 = vector.insert %b2, %bv1[2] : f32 into vector<8xf32>

        %bidx3 = affine.apply #micro_kernel_offset(%b_offset, %c3, %p)[%KC]
        %b3 = memref.load %b_packed[%bidx3] : memref<?xf32>
        %bv3 = vector.insert %b3, %bv2[3] : f32 into vector<8xf32>

        %bidx4 = affine.apply #micro_kernel_offset(%b_offset, %c4, %p)[%KC]
        %b4 = memref.load %b_packed[%bidx4] : memref<?xf32>
        %bv4 = vector.insert %b4, %bv3[4] : f32 into vector<8xf32>

        %bidx5 = affine.apply #micro_kernel_offset(%b_offset, %c5, %p)[%KC]
        %b5 = memref.load %b_packed[%bidx5] : memref<?xf32>
        %bv5 = vector.insert %b5, %bv4[5] : f32 into vector<8xf32>

        %bidx6 = affine.apply #micro_kernel_offset(%b_offset, %c6, %p)[%KC]
        %b6 = memref.load %b_packed[%bidx6] : memref<?xf32>
        %bv6 = vector.insert %b6, %bv5[6] : f32 into vector<8xf32>

        %bidx7 = affine.apply #micro_kernel_offset(%b_offset, %c7, %p)[%KC]
        %b7 = memref.load %b_packed[%bidx7] : memref<?xf32>
        %bvec = vector.insert %b7, %bv6[7] : f32 into vector<8xf32>

        %aidx0 = affine.apply #micro_kernel_offset(%a_offset, %c0, %p)[%KC]
        %a0 = memref.load %a_packed[%aidx0] : memref<?xf32>
        %a0v = vector.splat %a0 : vector<8xf32>
        %acc0_new = vector.fma %a0v, %bvec, %acc0 : vector<8xf32>

        %aidx1 = affine.apply #micro_kernel_offset(%a_offset, %c1, %p)[%KC]
        %a1 = memref.load %a_packed[%aidx1] : memref<?xf32>
        %a1v = vector.splat %a1 : vector<8xf32>
        %acc1_new = vector.fma %a1v, %bvec, %acc1 : vector<8xf32>

        %aidx2 = affine.apply #micro_kernel_offset(%a_offset, %c2, %p)[%KC]
        %a2 = memref.load %a_packed[%aidx2] : memref<?xf32>
        %a2v = vector.splat %a2 : vector<8xf32>
        %acc2_new = vector.fma %a2v, %bvec, %acc2 : vector<8xf32>

        %aidx3 = affine.apply #micro_kernel_offset(%a_offset, %c3, %p)[%KC]
        %a3 = memref.load %a_packed[%aidx3] : memref<?xf32>
        %a3v = vector.splat %a3 : vector<8xf32>
        %acc3_new = vector.fma %a3v, %bvec, %acc3 : vector<8xf32>

        scf.yield %acc0_new, %acc1_new, %acc2_new, %acc3_new
          : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
    }
    %base0 = affine.apply affine_map<(i)[NR]->(i*NR)>(%c0)[%NR]
    vector.store %acc0_fin, %c_sub[%base0] : memref<?xf32>, vector<8xf32>
    %base1 = affine.apply affine_map<(i)[NR]->(i*NR)>(%c1)[%NR]
    vector.store %acc1_fin, %c_sub[%base1] : memref<?xf32>, vector<8xf32>
    %base2 = affine.apply affine_map<(i)[NR]->(i*NR)>(%c2)[%NR]
    vector.store %acc2_fin, %c_sub[%base2] : memref<?xf32>, vector<8xf32>
    %base3 = affine.apply affine_map<(i)[NR]->(i*NR)>(%c3)[%NR]
    vector.store %acc3_fin, %c_sub[%base3] : memref<?xf32>, vector<8xf32>
    return
  }

  func.func @sgemm_blis_32(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    %t_start = func.call @rtclock() : () -> f64

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %M = memref.dim %a, %c0 : memref<?x?xf32>
    %N = memref.dim %c, %c1 : memref<?x?xf32>
    %K = memref.dim %a, %c1 : memref<?x?xf32>

    %NC = arith.constant 1024 : index
    %MC = arith.constant 256 : index
    %KC = arith.constant 128 : index

    %MR = arith.constant 4 : index
    %NR = arith.constant 8 : index

    %a_packed_size = arith.muli %MC, %KC : index
    %b_packed_size = arith.muli %KC, %NC : index

    %a_packed = memref.alloc(%a_packed_size) : memref<?xf32>
    %b_packed = memref.alloc(%b_packed_size) : memref<?xf32>

    %c_sub_size = arith.muli %MR, %NR: index
    %c_sub = memref.alloc(%c_sub_size) : memref<?xf32>

    affine.for %j_c = 0 to %N step 1024 { // NC
      affine.for %p_c = 0 to %K step 128 { // KC
        affine.for %i_c = 0 to %M step 256 { // MC
          func.call @pack_a(%a, %a_packed, %i_c, %p_c) : (memref<?x?xf32>, memref<?xf32>, index, index) -> ()
          func.call @pack_b(%b, %b_packed, %j_c, %p_c) : (memref<?x?xf32>, memref<?xf32>, index, index) -> ()

          affine.for %j_r = 0 to %NC step 8 { // NR
            affine.for %i_r = 0 to %MC step 4 { // MR
              %a_offset = affine.apply affine_map<(i_r)[KC]->(i_r*KC)>(%i_r)[%KC]
              %b_offset = affine.apply affine_map<(j_r)[KC]->(j_r*KC)>(%j_r)[%KC]

              func.call @micro_kernel(%a_packed, %b_packed, %c_sub, %a_offset, %b_offset) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index, index) -> ()

              affine.for %i = 0 to %MR {
                affine.for %j = 0 to %NR {
                  %c_sub_idx = affine.apply affine_map<(i,j)[NR]->(i*NR + j)>(%i, %j)[%NR]
                  %c_sub_val = memref.load %c_sub[%c_sub_idx] : memref<?xf32>

                  %i_global = affine.apply affine_map<(i_c,i_r,i)->(i_c+i_r+i)>(%i_c, %i_r, %i)
                  %j_global = affine.apply affine_map<(j_c,j_r,j)->(j_c+j_r+j)>(%j_c, %j_r, %j)

                  affine.if #cond_set_a(%i_global, %j_global)[%M, %N] {
                    %c_val = memref.load %c[%i_global, %j_global] : memref<?x?xf32>
                    %c_new = arith.addf %c_val, %c_sub_val : f32
                    memref.store %c_new, %c[%i_global, %j_global] : memref<?x?xf32>
                  }
                }
              }
            }
          }
        }
      }
    }

    memref.dealloc %c_sub : memref<?xf32>
    memref.dealloc %a_packed : memref<?xf32>
    memref.dealloc %b_packed : memref<?xf32>

    %t_end = func.call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
    func.return
  }

  func.func @main(){
    // Set up dims.
    %cM = arith.constant 1024 : index
    %cN = arith.constant 1536 : index
    %cK = arith.constant 8960 : index

    // Set Init Value.
    %cf1 = arith.constant 1.0 : f32
    %cf2 = arith.constant 2.0 : f32
    %c0 = arith.constant 0.0 : f32

    %A = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %B = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C = memref.alloc(%cM, %cN) : memref<?x?xf32>

    linalg.fill
    ins(%cf1 : f32)
    outs(%A:memref<?x?xf32>)

    linalg.fill
    ins(%cf2 : f32)
    outs(%B:memref<?x?xf32>)

    linalg.fill
    ins(%c0 : f32)
    outs(%C:memref<?x?xf32>)

    call @sgemm_blis_32(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    %i = arith.constant 0 : index
    %j = arith.constant 0 : index
    %val = memref.load %C[%i, %j] : memref<?x?xf32>
    vector.print %val : f32

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>
    return
  }
}
