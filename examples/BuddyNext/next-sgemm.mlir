// RUN: buddy-opt  %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -cse \
// RUN:   -lower-affine \
// RUN:   -convert-vector-to-scf \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @sgemm_vl_48(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index

    %m = memref.dim %a, %c0 : memref<?x?xf32>
    %n = memref.dim %c, %c1 : memref<?x?xf32>
    %k = memref.dim %a, %c1 : memref<?x?xf32>

    %step = arith.constant 48 : index

    affine.for %m_idx = 0 to %m step 8 {
      %m_idx_1 = arith.addi %m_idx, %c1 : index
      %m_idx_2 = arith.addi %m_idx, %c2 : index
      %m_idx_3 = arith.addi %m_idx, %c3 : index
      %m_idx_4 = arith.addi %m_idx, %c4 : index
      %m_idx_5 = arith.addi %m_idx, %c5 : index
      %m_idx_6 = arith.addi %m_idx, %c6 : index
      %m_idx_7 = arith.addi %m_idx, %c7 : index

      %n_body_bound_ = arith.subi %n, %step : index
      %n_body_bound = arith.addi %n_body_bound_, %c1 : index

      %n_iter_idx = scf.for %n_idx = %c0 to %n_body_bound step %step
          iter_args(%n_iter_idx_init = %c0) -> (index) {
      %sum_init = arith.constant dense<0.> : vector<48xf32>
      %sum_iter_vec_0, %sum_iter_vec_1, %sum_iter_vec_2, %sum_iter_vec_3,
      %sum_iter_vec_4, %sum_iter_vec_5, %sum_iter_vec_6, %sum_iter_vec_7
          = affine.for %k_idx = 0 to %k
          iter_args(%sum_vec_0 = %sum_init,
                      %sum_vec_1 = %sum_init,
                      %sum_vec_2 = %sum_init,
                      %sum_vec_3 = %sum_init,
                      %sum_vec_4 = %sum_init,
                      %sum_vec_5 = %sum_init,
                      %sum_vec_6 = %sum_init,
                      %sum_vec_7 = %sum_init
                      )
          -> (vector<48xf32>, vector<48xf32>, vector<48xf32>, vector<48xf32>,
              vector<48xf32>, vector<48xf32>, vector<48xf32>, vector<48xf32>) {
          %a_ele_0 = memref.load %a[%m_idx, %k_idx] : memref<?x?xf32>
          %a_ele_1 = memref.load %a[%m_idx_1, %k_idx] : memref<?x?xf32>
          %a_ele_2 = memref.load %a[%m_idx_2, %k_idx] : memref<?x?xf32>
          %a_ele_3 = memref.load %a[%m_idx_3, %k_idx] : memref<?x?xf32>
          %a_ele_4 = memref.load %a[%m_idx_4, %k_idx] : memref<?x?xf32>
          %a_ele_5 = memref.load %a[%m_idx_5, %k_idx] : memref<?x?xf32>
          %a_ele_6 = memref.load %a[%m_idx_6, %k_idx] : memref<?x?xf32>
          %a_ele_7 = memref.load %a[%m_idx_7, %k_idx] : memref<?x?xf32>
          %a_vec_0 = vector.broadcast %a_ele_0 : f32 to vector<48xf32>
          %a_vec_1 = vector.broadcast %a_ele_1 : f32 to vector<48xf32>
          %a_vec_2 = vector.broadcast %a_ele_2 : f32 to vector<48xf32>
          %a_vec_3 = vector.broadcast %a_ele_3 : f32 to vector<48xf32>
          %a_vec_4 = vector.broadcast %a_ele_4 : f32 to vector<48xf32>
          %a_vec_5 = vector.broadcast %a_ele_5 : f32 to vector<48xf32>
          %a_vec_6 = vector.broadcast %a_ele_6 : f32 to vector<48xf32>
          %a_vec_7 = vector.broadcast %a_ele_7 : f32 to vector<48xf32>
          %b_vec = vector.load %b[%k_idx, %n_idx] : memref<?x?xf32>, vector<48xf32>
          %res_sum_vec_0 = vector.fma %a_vec_0, %b_vec, %sum_vec_0 : vector<48xf32>
          %res_sum_vec_1 = vector.fma %a_vec_1, %b_vec, %sum_vec_1 : vector<48xf32>
          %res_sum_vec_2 = vector.fma %a_vec_2, %b_vec, %sum_vec_2 : vector<48xf32>
          %res_sum_vec_3 = vector.fma %a_vec_3, %b_vec, %sum_vec_3 : vector<48xf32>
          %res_sum_vec_4 = vector.fma %a_vec_4, %b_vec, %sum_vec_4 : vector<48xf32>
          %res_sum_vec_5 = vector.fma %a_vec_5, %b_vec, %sum_vec_5 : vector<48xf32>
          %res_sum_vec_6 = vector.fma %a_vec_6, %b_vec, %sum_vec_6 : vector<48xf32>
          %res_sum_vec_7 = vector.fma %a_vec_7, %b_vec, %sum_vec_7 : vector<48xf32>
          affine.yield %res_sum_vec_0, %res_sum_vec_1, %res_sum_vec_2, %res_sum_vec_3,
                      %res_sum_vec_4, %res_sum_vec_5, %res_sum_vec_6, %res_sum_vec_7
              : vector<48xf32>, vector<48xf32>, vector<48xf32>, vector<48xf32>,
              vector<48xf32>, vector<48xf32>, vector<48xf32>, vector<48xf32>
      }
      vector.store %sum_iter_vec_0, %c[%m_idx, %n_idx] : memref<?x?xf32>, vector<48xf32>
      vector.store %sum_iter_vec_1, %c[%m_idx_1, %n_idx] : memref<?x?xf32>, vector<48xf32>
      vector.store %sum_iter_vec_2, %c[%m_idx_2, %n_idx] : memref<?x?xf32>, vector<48xf32>
      vector.store %sum_iter_vec_3, %c[%m_idx_3, %n_idx] : memref<?x?xf32>, vector<48xf32>
      vector.store %sum_iter_vec_4, %c[%m_idx_4, %n_idx] : memref<?x?xf32>, vector<48xf32>
      vector.store %sum_iter_vec_5, %c[%m_idx_5, %n_idx] : memref<?x?xf32>, vector<48xf32>
      vector.store %sum_iter_vec_6, %c[%m_idx_6, %n_idx] : memref<?x?xf32>, vector<48xf32>
      vector.store %sum_iter_vec_7, %c[%m_idx_7, %n_idx] : memref<?x?xf32>, vector<48xf32>
      %k_next = arith.addi %n_idx, %step : index
      scf.yield %k_next : index
      }
      // TODO: Add tail processing for both horizontal and vertical.
      scf.for %n_idx = %n_iter_idx to %n step %c1 {
        %sum_init = arith.constant 0. : f32
        %sum_iter_0, %sum_iter_1, %sum_iter_2, %sum_iter_3,
        %sum_iter_4, %sum_iter_5, %sum_iter_6, %sum_iter_7
            = affine.for %k_idx = 0 to %k
            iter_args(%sum_0 = %sum_init,
                      %sum_1 = %sum_init,
                      %sum_2 = %sum_init,
                      %sum_3 = %sum_init,
                      %sum_4 = %sum_init,
                      %sum_5 = %sum_init,
                      %sum_6 = %sum_init,
                      %sum_7 = %sum_init
                      ) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
          %a_ele_0 = memref.load %a[%m_idx, %k_idx] : memref<?x?xf32>
          %a_ele_1 = memref.load %a[%m_idx_1, %k_idx] : memref<?x?xf32>
          %a_ele_2 = memref.load %a[%m_idx_2, %k_idx] : memref<?x?xf32>
          %a_ele_3 = memref.load %a[%m_idx_3, %k_idx] : memref<?x?xf32>
          %a_ele_4 = memref.load %a[%m_idx_4, %k_idx] : memref<?x?xf32>
          %a_ele_5 = memref.load %a[%m_idx_5, %k_idx] : memref<?x?xf32>
          %a_ele_6 = memref.load %a[%m_idx_6, %k_idx] : memref<?x?xf32>
          %a_ele_7 = memref.load %a[%m_idx_7, %k_idx] : memref<?x?xf32>

          %b_ele = memref.load %b[%k_idx, %n_idx] : memref<?x?xf32>

          %tmp_ele_0 = arith.mulf %a_ele_0, %b_ele : f32
          %tmp_ele_1 = arith.mulf %a_ele_1, %b_ele : f32
          %tmp_ele_2 = arith.mulf %a_ele_2, %b_ele : f32
          %tmp_ele_3 = arith.mulf %a_ele_3, %b_ele : f32
          %tmp_ele_4 = arith.mulf %a_ele_4, %b_ele : f32
          %tmp_ele_5 = arith.mulf %a_ele_5, %b_ele : f32
          %tmp_ele_6 = arith.mulf %a_ele_6, %b_ele : f32
          %tmp_ele_7 = arith.mulf %a_ele_7, %b_ele : f32

          %res_sum_0 = arith.addf %tmp_ele_0, %sum_0 : f32
          %res_sum_1 = arith.addf %tmp_ele_1, %sum_1 : f32
          %res_sum_2 = arith.addf %tmp_ele_2, %sum_2 : f32
          %res_sum_3 = arith.addf %tmp_ele_3, %sum_3 : f32
          %res_sum_4 = arith.addf %tmp_ele_4, %sum_4 : f32
          %res_sum_5 = arith.addf %tmp_ele_5, %sum_5 : f32
          %res_sum_6 = arith.addf %tmp_ele_6, %sum_6 : f32
          %res_sum_7 = arith.addf %tmp_ele_7, %sum_7 : f32

          affine.yield %res_sum_0,
                       %res_sum_1,
                       %res_sum_2,
                       %res_sum_3,
                       %res_sum_4,
                       %res_sum_5,
                       %res_sum_6,
                       %res_sum_7 : f32, f32, f32, f32, f32, f32, f32, f32
        }
        memref.store %sum_iter_0, %c[%m_idx, %n_idx] : memref<?x?xf32>
        memref.store %sum_iter_1, %c[%m_idx_1, %n_idx] : memref<?x?xf32>
        memref.store %sum_iter_2, %c[%m_idx_2, %n_idx] : memref<?x?xf32>
        memref.store %sum_iter_3, %c[%m_idx_3, %n_idx] : memref<?x?xf32>
        memref.store %sum_iter_4, %c[%m_idx_4, %n_idx] : memref<?x?xf32>
        memref.store %sum_iter_5, %c[%m_idx_5, %n_idx] : memref<?x?xf32>
        memref.store %sum_iter_6, %c[%m_idx_6, %n_idx] : memref<?x?xf32>
        memref.store %sum_iter_7, %c[%m_idx_7, %n_idx] : memref<?x?xf32>
      }
    }
    return
  }

  func.func @main(){
    // Set up dims.
    %cM = arith.constant 40 : index
    %cN = arith.constant 14336 : index
    %cK = arith.constant 4096 : index

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

    call @sgemm_vl_48(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>
    return
  }
}

  // All the elements of the MemRef are the same,
  // only check the first line to verify the correctness.
  // CHECK: Unranked Memref base@ = {{.*}}
  // CHECK-NEXT: [
  // CHECK-SAME: [8192{{(, 8192)*}}],
