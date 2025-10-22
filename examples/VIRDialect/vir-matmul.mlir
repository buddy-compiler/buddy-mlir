// RUN: buddy-opt %s \
// RUN: | FileCheck %s

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  // Set up dims.
  %m = arith.constant 1024 : index
  %n = arith.constant 1024 : index
  %k = arith.constant 1024 : index

  // Set Init Value.
  %cf1 = arith.constant 1.0 : f32

  %a = memref.alloc(%m, %k) : memref<?x?xf32>
  %b = memref.alloc(%k, %n) : memref<?x?xf32>
  %c = memref.alloc(%m, %n) : memref<?x?xf32>

  linalg.fill ins(%cf1 : f32) outs(%a:memref<?x?xf32>)
  linalg.fill ins(%cf1 : f32) outs(%b:memref<?x?xf32>)
  linalg.fill ins(%cf1 : f32) outs(%c:memref<?x?xf32>)

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  // %a = m x k | %b = k x n | %c = m x n
  // Unroll the M dimension by 4.
  affine.for %m_idx_0 = 0 to %m step 4 {
    %m_idx_1 = arith.addi %m_idx_0, %c1 : index
    %m_idx_2 = arith.addi %m_idx_0, %c2 : index
    %m_idx_3 = arith.addi %m_idx_0, %c3 : index
    // CHECK: vir.set_vl {{.*}}
    // Use SetVL region to compute an entire row of the target matrix.
    vir.set_vl %n : index {
      %sum_init_0 = vir.constant { value = 0.0 : f32 } : !vir.vec<?xf32>
      %sum_init_1 = vir.constant { value = 0.0 : f32 } : !vir.vec<?xf32>
      %sum_init_2 = vir.constant { value = 0.0 : f32 } : !vir.vec<?xf32>
      %sum_init_3 = vir.constant { value = 0.0 : f32 } : !vir.vec<?xf32>
      // Innermost loop
      %sum_iter_vec_0, %sum_iter_vec_1, %sum_iter_vec_2, %sum_iter_vec_3
          = affine.for %k_idx = 0 to %k
          iter_args(%sum_vec_0 = %sum_init_0, %sum_vec_1 = %sum_init_1,
                    %sum_vec_2 = %sum_init_2, %sum_vec_3 = %sum_init_3)
          -> (!vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>) {
        %a_ele_0 = memref.load %a[%m_idx_0, %k_idx] : memref<?x?xf32>
        %a_ele_1 = memref.load %a[%m_idx_1, %k_idx] : memref<?x?xf32>
        %a_ele_2 = memref.load %a[%m_idx_2, %k_idx] : memref<?x?xf32>
        %a_ele_3 = memref.load %a[%m_idx_3, %k_idx] : memref<?x?xf32>

        %a_vec_0 = vir.broadcast %a_ele_0 : f32 -> !vir.vec<?xf32>
        %a_vec_1 = vir.broadcast %a_ele_1 : f32 -> !vir.vec<?xf32>
        %a_vec_2 = vir.broadcast %a_ele_2 : f32 -> !vir.vec<?xf32>
        %a_vec_3 = vir.broadcast %a_ele_3 : f32 -> !vir.vec<?xf32>

        %b_vec = vir.load %b[%k_idx, %c0] : memref<?x?xf32> -> !vir.vec<?xf32>

        %res_sum_vec_0 = vir.fma %a_vec_0, %b_vec, %sum_vec_0 : !vir.vec<?xf32>
        %res_sum_vec_1 = vir.fma %a_vec_1, %b_vec, %sum_vec_1 : !vir.vec<?xf32>
        %res_sum_vec_2 = vir.fma %a_vec_2, %b_vec, %sum_vec_2 : !vir.vec<?xf32>
        %res_sum_vec_3 = vir.fma %a_vec_3, %b_vec, %sum_vec_3 : !vir.vec<?xf32>

        affine.yield %res_sum_vec_0, %res_sum_vec_1, %res_sum_vec_2, %res_sum_vec_3
            : !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>, !vir.vec<?xf32>
      }
      // Out of the innermost loop
      vir.store %sum_iter_vec_0, %c[%m_idx_0, %c0] : !vir.vec<?xf32> -> memref<?x?xf32>
      vir.store %sum_iter_vec_1, %c[%m_idx_1, %c0] : !vir.vec<?xf32> -> memref<?x?xf32>
      vir.store %sum_iter_vec_2, %c[%m_idx_2, %c0] : !vir.vec<?xf32> -> memref<?x?xf32>
      vir.store %sum_iter_vec_3, %c[%m_idx_3, %c0] : !vir.vec<?xf32> -> memref<?x?xf32>
      vector.yield
    }
  }

  %print_mem =  memref.cast %c : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()

  return
}
