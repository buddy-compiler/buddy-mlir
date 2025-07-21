// RUN: buddy-opt %s \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf -convert-scf-to-cf \
// RUN:     -convert-cf-to-llvm \
// RUN:     -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=i32 \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

memref.global "private" @gv : memref<4x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.],
                                                       [30., 31., 32., 33.]]>

memref.global "private" @gv_pat_1 : memref<10xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9.]>
memref.global "private" @gv_pat_2 : memref<10xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9.]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() -> i32 {
  %mem = memref.get_global @gv : memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %sum_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
  %sum = affine.for %i = 0 to 3 iter_args(%sum_iter = %sum_0) -> (vector<4xf32>) {
    %load_vec1 = vector.load %mem[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
    %load_vec2 = vector.load %mem[%i, %c0] : memref<4x4xf32>, vector<4xf32>
    %sum_next = vector.fma %load_vec1, %load_vec2, %sum_iter : vector<4xf32>
    affine.yield %sum_next : vector<4xf32>
  }
  // CHECK: ( 0, 33, 72, 117 )
  vector.print %sum : vector<4xf32>

  // ---------------------------------------------------------------------------
  // Iteration Pattern 1
  // Main Vector Loop + Scalar Remainder + Fixed Vector Type
  // ---------------------------------------------------------------------------

  // 1. Get the total length of the workload.
  %mem_pat_1 = memref.get_global @gv_pat_1 : memref<10xf32>
  %print_mem_pat_1 = memref.cast %mem_pat_1 : memref<10xf32> to memref<*xf32>
  %vl_total_pat_1 = memref.dim %mem_pat_1, %c0 : memref<10xf32>

  // 2. Set the iteration step (vector size).
  %vl_step_pat_1 = arith.constant 4 : index

  // 3. Calculate the upper bound for vectorized processing
  // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
  // - Add 1 to ensure the final loop runs when the workload length is divisible
  //   by the vector size.
  %vl_upbound_pat_1_ = arith.subi %vl_total_pat_1, %vl_step_pat_1 : index
  %vl_upbound_pat_1 = arith.addi %vl_upbound_pat_1_, %c1 : index

  // 4. Perform the vectorization body.
  %iter_idx_pat_1 = scf.for %i = %c0 to %vl_upbound_pat_1 step %vl_step_pat_1
      iter_args(%iter_init = %c0) -> (index) {
    %load_vec1 = vector.load %mem_pat_1[%i] : memref<10xf32>, vector<4xf32>
    %load_vec2 = vector.load %mem_pat_1[%i] : memref<10xf32>, vector<4xf32>
    %res = arith.addf %load_vec1, %load_vec2 : vector<4xf32>
    vector.store %res, %mem_pat_1[%i] : memref<10xf32>, vector<4xf32>
    %i_next = arith.addi %i, %vl_step_pat_1 : index
    scf.yield %i_next : index
  }
  // CHECK: [0,  2,  4,  6,  8,  10,  12,  14,  8,  9]
  call @printMemrefF32(%print_mem_pat_1) : (memref<*xf32>) -> ()

  // 5. Process the remainder of the elements with scalar operations.
  scf.for %i = %iter_idx_pat_1 to %vl_total_pat_1 step %c1 {
    %ele1 = memref.load %mem_pat_1[%i] : memref<10xf32>
    %ele2 = memref.load %mem_pat_1[%i] : memref<10xf32>
    %res = arith.addf %ele1, %ele2 : f32
    memref.store %res, %mem_pat_1[%i] : memref<10xf32>
  }
  // CHECK: [0,  2,  4,  6,  8,  10,  12,  14,  16,  18]
  call @printMemrefF32(%print_mem_pat_1) : (memref<*xf32>) -> ()

  // ---------------------------------------------------------------------------
  // Iteration Pattern 2
  // Main Vector Loop + Masked Vector Remainder + Fixed Vector Type
  // ---------------------------------------------------------------------------

  // 1. Get the total length of the workload.
  %mem_pat_2 = memref.get_global @gv_pat_2 : memref<10xf32>
  %print_mem_pat_2 = memref.cast %mem_pat_2 : memref<10xf32> to memref<*xf32>
  %vl_total_pat_2 = memref.dim %mem_pat_2, %c0 : memref<10xf32>

  // 2. Set the iteration step (vector size).
  %vl_step_pat_2 = arith.constant 4 : index

  // 3. Calculate the upper bound for vectorized processing
  // - Subtract `vl_step` is to avoid overflow at the vectorization tail.
  // - Add 1 to ensure the final loop runs when the workload length is divisible
  //   by the vector size.
  %vl_upbound_pat_2_ = arith.subi %vl_total_pat_2, %vl_step_pat_2 : index
  %vl_upbound_pat_2 = arith.addi %vl_upbound_pat_2_, %c1 : index

  // 4. Perform the vectorization body.
  %iter_idx_pat_2 = scf.for %i = %c0 to %vl_upbound_pat_2 step %vl_step_pat_2
      iter_args(%iter_init = %c0) -> (index) {
    %load_vec1 = vector.load %mem_pat_2[%i] : memref<10xf32>, vector<4xf32>
    %load_vec2 = vector.load %mem_pat_2[%i] : memref<10xf32>, vector<4xf32>
    %res = arith.addf %load_vec1, %load_vec2 : vector<4xf32>
    vector.store %res, %mem_pat_2[%i] : memref<10xf32>, vector<4xf32>
    %i_next = arith.addi %i, %vl_step_pat_1 : index
    scf.yield %i_next : index
  }
  // CHECK: [0,  2,  4,  6,  8,  10,  12,  14,  8,  9]
  call @printMemrefF32(%print_mem_pat_2) : (memref<*xf32>) -> ()

  // 5. Compute the tail size and create mask and pass-through vector for the
  //    remaining elements.
  %tail_size_pat_2 = arith.subi %vl_total_pat_2, %iter_idx_pat_2 : index
  %mask_pat_2 = vector.create_mask %tail_size_pat_2 : vector<4xi1>
  %pass_thr_vec = arith.constant dense<0.> : vector<4xf32>

  // 6. Process the remaining elements using masked vector operations.
  %ele1 = vector.maskedload %mem_pat_2[%iter_idx_pat_2], %mask_pat_2, %pass_thr_vec : memref<10xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  %ele2 = vector.maskedload %mem_pat_2[%iter_idx_pat_2], %mask_pat_2, %pass_thr_vec : memref<10xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  %res = arith.addf %ele1, %ele2 : vector<4xf32>
  vector.maskedstore %mem_pat_2[%iter_idx_pat_2], %mask_pat_2, %res : memref<10xf32>, vector<4xi1>, vector<4xf32>
  // CHECK: [0,  2,  4,  6,  8,  10,  12,  14,  16,  18]
  call @printMemrefF32(%print_mem_pat_2) : (memref<*xf32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
