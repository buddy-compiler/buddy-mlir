// RUN: buddy-opt %s \
// RUN:   -convert-linalg-to-loops \
// RUN:   -cse \
// RUN:   -lower-affine \
// RUN:   -convert-vector-to-scf \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-cf-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64

  // dot kernel，Ret = dot(A_row, B_col) + C[ir0, ir1]
  func.func @dot_add(%A: memref<?x?xf32>, %a_row: index, %B: memref<?x?xf32>, %b_col: index, %K: index, %C: memref<?x?xf32>, %c_row: index, %c_col: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %vn = arith.constant 8 : index
    %zero_f = arith.constant 0.0 : f32

    %vec_iters = arith.divui %K, %vn : index
    %vec_limit = arith.muli %vec_iters, %vn : index

    %acc_mem = memref.alloc() : memref<8xf32>
    scf.for %i = %c0 to %vn step %c1 {
      memref.store %zero_f, %acc_mem[%i] : memref<8xf32>
    }
    scf.for %kk = %c0 to %vec_limit step %vn {
      %avec = vector.load %A[%a_row, %kk] : memref<?x?xf32>, vector<8xf32>
      %bvec = vector.load %B[%kk, %b_col] : memref<?x?xf32>, vector<8xf32>
      %prev = vector.load %acc_mem[%c0] : memref<8xf32>, vector<8xf32>
      %sumvec = vector.fma %avec, %bvec, %prev : vector<8xf32>
      vector.store %sumvec, %acc_mem[%c0] : memref<8xf32>, vector<8xf32>
    }
    %acc_scalar = scf.for %i = %c0 to %vn step %c1 iter_args(%s = %zero_f) -> (f32) {
      %vitem = memref.load %acc_mem[%i] : memref<8xf32>
      %s2 = arith.addf %s, %vitem : f32
      scf.yield %s2 : f32
    }
    %tail_sum = scf.for %kk = %vec_limit to %K step %c1 iter_args(%s = %acc_scalar) -> (f32) {
      %a = memref.load %A[%a_row, %kk] : memref<?x?xf32>
      %b = memref.load %B[%kk, %b_col] : memref<?x?xf32>
      %prod = arith.mulf %a, %b : f32
      %s2 = arith.addf %s, %prod : f32
      scf.yield %s2 : f32
    }
    memref.dealloc %acc_mem : memref<8xf32>
    // Add C[ir0, ir1]
    %c_orig = memref.load %C[%c_row, %c_col] : memref<?x?xf32>
    %result = arith.addf %tail_sum, %c_orig : f32
    func.return %result : f32
  }

  func.func @mul_mat_one_chunk(
    %A: memref<?x?xf32>,
    %B: memref<?x?xf32>,
    %C: memref<?x?xf32>,
    %num_rows_per_vec_dot: index,
    %ir0_start: index, %ir0_end: index,
    %ir1_start: index, %ir1_end: index,
    %K: index
  ) {
    %blck_0 = arith.constant 16 : index
    %blck_1 = arith.constant 16 : index
    %C0=arith.constant 0 : index
    %C1=arith.constant 1 : index

    scf.for %iir1 = %ir1_start to %ir1_end step %blck_1 {
      %iir1_end_tmp = arith.addi %iir1, %blck_1 : index
      %iir1_lt = arith.cmpi slt, %iir1_end_tmp, %ir1_end : index
      %iir1_end = arith.select %iir1_lt, %iir1_end_tmp, %ir1_end : index
      %nj = arith.subi %iir1_end, %iir1 : index

      scf.parallel (%iir0) = (%ir0_start) to (%ir0_end) step (%blck_0) {
        %iir0_end_tmp = arith.addi %iir0, %blck_0 : index
        %iir0_lt = arith.cmpi slt, %iir0_end_tmp, %ir0_end : index
        %iir0_end = arith.select %iir0_lt, %iir0_end_tmp, %ir0_end : index
        scf.for %ir1 = %iir1 to %iir1_end step %num_rows_per_vec_dot {
          scf.for %ir0 = %iir0 to %iir0_end step %num_rows_per_vec_dot {
            %dotval = func.call @dot_add(%A, %ir0, %B, %ir1, %K, %C, %ir0, %ir1)
              : (memref<?x?xf32>, index, memref<?x?xf32>, index, index, memref<?x?xf32>, index, index) -> f32
            memref.store %dotval, %C[%ir0, %ir1] : memref<?x?xf32>
          }
        }
      }
    }
    func.return
  }

  // Top matmul
  func.func @ggml_mul_mat_mlir(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
    %zero = arith.constant 0 : index
    %one  = arith.constant 1 : index

    %M = memref.dim %A, %zero : memref<?x?xf32>
    %K = memref.dim %A, %one : memref<?x?xf32>
    %N = memref.dim %B, %one : memref<?x?xf32>

    %chunk16 = arith.constant 16 : index
    %chunk64 = arith.constant 64 : index

    %cond_nr0_one = arith.cmpi eq, %M, %one : index
    %cond_nr1_one = arith.cmpi eq, %N, %one : index
    %chunk_size_tmp = arith.select %cond_nr0_one, %chunk64, %chunk16 : index
    %chunk_size = arith.select %cond_nr1_one, %chunk64, %chunk_size_tmp : index

    %tmp0 = arith.addi %M, %chunk_size : index
    %tmp0_1 = arith.subi %tmp0, %one : index
    %nchunk0 = arith.divui %tmp0_1, %chunk_size : index

    %tmp1 = arith.addi %N, %chunk_size : index
    %tmp1_1 = arith.subi %tmp1, %one : index
    %nchunk1 = arith.divui %tmp1_1, %chunk_size : index

    %dr0_tmp = arith.addi %M, %nchunk0 : index
    %dr0_tmp1 = arith.subi %dr0_tmp, %one : index
    %dr0 = arith.divui %dr0_tmp1, %nchunk0 : index

    %dr1_tmp = arith.addi %N, %nchunk1 : index
    %dr1_tmp1 = arith.subi %dr1_tmp, %one : index
    %dr1 = arith.divui %dr1_tmp1, %nchunk1 : index

    %total_chunks = arith.muli %nchunk0, %nchunk1 : index

    scf.parallel (%current_chunk) = (%zero) to (%total_chunks) step (%one) {
      %ith0 = arith.remui %current_chunk, %nchunk0 : index
      %ith1 = arith.divui %current_chunk, %nchunk0 : index

      %ir0_start = arith.muli %dr0, %ith0 : index
      %ir0_end_tmp = arith.addi %ir0_start, %dr0 : index
      %ir0_lt = arith.cmpi slt, %ir0_end_tmp, %M : index
      %ir0_end = arith.select %ir0_lt, %ir0_end_tmp, %M : index

      %ir1_start = arith.muli %dr1, %ith1 : index
      %ir1_end_tmp = arith.addi %ir1_start, %dr1 : index
      %ir1_lt = arith.cmpi slt, %ir1_end_tmp, %N : index
      %ir1_end = arith.select %ir1_lt, %ir1_end_tmp, %N : index

      %num_rows_per_vec_dot = arith.constant 1 : index
      func.call @mul_mat_one_chunk(%A, %B, %C, %num_rows_per_vec_dot, %ir0_start, %ir0_end, %ir1_start, %ir1_end, %K)
        : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, index, index, index) -> ()
    }
    func.return
  }

  // Verify main function
  func.func @main() {
    %cM = arith.constant 34 : index
    %cN = arith.constant 29 : index
    %cK = arith.constant 1536 : index
    %cf0_32 = arith.constant 0.0 : f32
    %cf1_32 = arith.constant 1.0 : f32
    %cf2_32 = arith.constant 2.0 : f32

    %A_f32 = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %B_f32 = memref.alloc(%cK, %cN) : memref<?x?xf32>
    %C_f32 = memref.alloc(%cM, %cN) : memref<?x?xf32>

    linalg.fill ins(%cf1_32 : f32) outs(%A_f32 : memref<?x?xf32>)
    linalg.fill ins(%cf2_32 : f32) outs(%B_f32 : memref<?x?xf32>)
    linalg.fill ins(%cf1_32 : f32) outs(%C_f32 : memref<?x?xf32>)

    %t_start = call @rtclock() : () -> f64
    call @ggml_mul_mat_mlir(%A_f32, %B_f32, %C_f32) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    %t_end = call @rtclock() : () -> f64
    %time = arith.subf %t_end, %t_start : f64

    // %print_C_f32 = memref.cast %C_f32 : memref<?x?xf32> to memref<*xf32>
    // call @printMemrefF32(%print_C_f32) : (memref<*xf32>) -> ()

    vector.print %time : f64
    // CHECK: {{[0-9]+\.[0-9]+}}
    memref.dealloc %C_f32 : memref<?x?xf32>
    memref.dealloc %B_f32 : memref<?x?xf32>
    memref.dealloc %A_f32 : memref<?x?xf32>
    return
  }
}
