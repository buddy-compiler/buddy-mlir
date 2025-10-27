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
  func.func private @rtclock() -> f64

  func.func @sgemm_vl_32(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>) {
    %t_start = call @rtclock() : () -> f64

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index

    %unroll = arith.constant 8 : index
    %sum_init = arith.constant dense<0.> : vector<32xf32>

    %m = memref.dim %a, %c0 : memref<?x?xf32>
    %n = memref.dim %c, %c1 : memref<?x?xf32>
    %k = memref.dim %a, %c1 : memref<?x?xf32>

    %step = arith.constant 32 : index
    %k_body_bound_ = arith.subi %k, %step : index
    %k_body_bound = arith.addi %k_body_bound_, %c1 : index

    scf.parallel (%m_idx) = (%c0) to (%m) step (%unroll) {
      %m_idx_1 = arith.addi %m_idx, %c1 : index
      %m_idx_2 = arith.addi %m_idx, %c2 : index
      %m_idx_3 = arith.addi %m_idx, %c3 : index
      %m_idx_4 = arith.addi %m_idx, %c4 : index
      %m_idx_5 = arith.addi %m_idx, %c5 : index
      %m_idx_6 = arith.addi %m_idx, %c6 : index
      %m_idx_7 = arith.addi %m_idx, %c7 : index

      scf.for %n_idx = %c0 to %n step %c1 {
        %tmp_iter_0, %tmp_iter_1, %tmp_iter_2, %tmp_iter_3, 
        %tmp_iter_4, %tmp_iter_5, %tmp_iter_6, %tmp_iter_7, %k_idx_iter
            = scf.for %k_idx = %c0 to %k_body_bound step %step
            iter_args(%sum_vec_0 = %sum_init,
                        %sum_vec_1 = %sum_init,
                        %sum_vec_2 = %sum_init,
                        %sum_vec_3 = %sum_init,
                        %sum_vec_4 = %sum_init,
                        %sum_vec_5 = %sum_init,
                        %sum_vec_6 = %sum_init,
                        %sum_vec_7 = %sum_init, 
                        %idx = %c0
                        )
            -> (vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>,
                vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, index) {
            %a_vec_0 = vector.load %a[%m_idx, %k_idx] : memref<?x?xf32>, vector<32xf32>
            %a_vec_1 = vector.load %a[%m_idx_1, %k_idx] : memref<?x?xf32>, vector<32xf32>
            %a_vec_2 = vector.load %a[%m_idx_2, %k_idx] : memref<?x?xf32>, vector<32xf32>
            %a_vec_3 = vector.load %a[%m_idx_3, %k_idx] : memref<?x?xf32>, vector<32xf32>
            %a_vec_4 = vector.load %a[%m_idx_4, %k_idx] : memref<?x?xf32>, vector<32xf32>
            %a_vec_5 = vector.load %a[%m_idx_5, %k_idx] : memref<?x?xf32>, vector<32xf32>
            %a_vec_6 = vector.load %a[%m_idx_6, %k_idx] : memref<?x?xf32>, vector<32xf32>
            %a_vec_7 = vector.load %a[%m_idx_7, %k_idx] : memref<?x?xf32>, vector<32xf32>
            %b_vec = vector.load %b[%n_idx, %k_idx] : memref<?x?xf32>, vector<32xf32>
            %res_sum_vec_0 = vector.fma %a_vec_0, %b_vec, %sum_vec_0 : vector<32xf32>
            %res_sum_vec_1 = vector.fma %a_vec_1, %b_vec, %sum_vec_1 : vector<32xf32>
            %res_sum_vec_2 = vector.fma %a_vec_2, %b_vec, %sum_vec_2 : vector<32xf32>
            %res_sum_vec_3 = vector.fma %a_vec_3, %b_vec, %sum_vec_3 : vector<32xf32>
            %res_sum_vec_4 = vector.fma %a_vec_4, %b_vec, %sum_vec_4 : vector<32xf32>
            %res_sum_vec_5 = vector.fma %a_vec_5, %b_vec, %sum_vec_5 : vector<32xf32>
            %res_sum_vec_6 = vector.fma %a_vec_6, %b_vec, %sum_vec_6 : vector<32xf32>
            %res_sum_vec_7 = vector.fma %a_vec_7, %b_vec, %sum_vec_7 : vector<32xf32>

            %k_next = arith.addi %k_idx, %step : index
            scf.yield %res_sum_vec_0, %res_sum_vec_1, %res_sum_vec_2, %res_sum_vec_3,
                      %res_sum_vec_4, %res_sum_vec_5, %res_sum_vec_6, %res_sum_vec_7, %k_next 
                : vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, 
                vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, index
        }

        %tmp_sum_0 = vector.reduction <add>, %tmp_iter_0, fastmath<reassoc> : vector<32xf32> into f32
        %tmp_sum_1 = vector.reduction <add>, %tmp_iter_1, fastmath<reassoc> : vector<32xf32> into f32
        %tmp_sum_2 = vector.reduction <add>, %tmp_iter_2, fastmath<reassoc> : vector<32xf32> into f32
        %tmp_sum_3 = vector.reduction <add>, %tmp_iter_3, fastmath<reassoc> : vector<32xf32> into f32
        %tmp_sum_4 = vector.reduction <add>, %tmp_iter_4, fastmath<reassoc> : vector<32xf32> into f32
        %tmp_sum_5 = vector.reduction <add>, %tmp_iter_5, fastmath<reassoc> : vector<32xf32> into f32
        %tmp_sum_6 = vector.reduction <add>, %tmp_iter_6, fastmath<reassoc> : vector<32xf32> into f32
        %tmp_sum_7 = vector.reduction <add>, %tmp_iter_7, fastmath<reassoc> : vector<32xf32> into f32

        %sum_iter_0, %sum_iter_1, %sum_iter_2, %sum_iter_3,
        %sum_iter_4, %sum_iter_5, %sum_iter_6, %sum_iter_7
            = scf.for %k_idx = %k_idx_iter to %k step %c1
            iter_args(%sum_0 = %tmp_sum_0,
                      %sum_1 = %tmp_sum_1,
                      %sum_2 = %tmp_sum_2,
                      %sum_3 = %tmp_sum_3,
                      %sum_4 = %tmp_sum_4,
                      %sum_5 = %tmp_sum_5,
                      %sum_6 = %tmp_sum_6,
                      %sum_7 = %tmp_sum_7
                      ) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
          %a_ele_0 = memref.load %a[%m_idx, %k_idx] : memref<?x?xf32>
          %a_ele_1 = memref.load %a[%m_idx_1, %k_idx] : memref<?x?xf32>
          %a_ele_2 = memref.load %a[%m_idx_2, %k_idx] : memref<?x?xf32>
          %a_ele_3 = memref.load %a[%m_idx_3, %k_idx] : memref<?x?xf32>
          %a_ele_4 = memref.load %a[%m_idx_4, %k_idx] : memref<?x?xf32>
          %a_ele_5 = memref.load %a[%m_idx_5, %k_idx] : memref<?x?xf32>
          %a_ele_6 = memref.load %a[%m_idx_6, %k_idx] : memref<?x?xf32>
          %a_ele_7 = memref.load %a[%m_idx_7, %k_idx] : memref<?x?xf32>

          %b_ele = memref.load %b[%n_idx, %k_idx] : memref<?x?xf32>

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

          scf.yield %res_sum_0,
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
      // TODO: Add tail processing for both horizontal and vertical.
    }

    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
    return
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
    %B = memref.alloc(%cN, %cK) : memref<?x?xf32>
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

    call @sgemm_vl_32(%A, %B, %C) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    // %print_C = memref.cast %C : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    memref.dealloc %C : memref<?x?xf32>
    memref.dealloc %B : memref<?x?xf32>
    memref.dealloc %A : memref<?x?xf32>
    return
  }
}
