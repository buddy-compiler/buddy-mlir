memref.global "private" @gv : memref<10xf32> = dense<[0. , 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9.]>

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() -> i32 {
  %c0 = arith.constant 0 : index

  // ---------------------------------------------------------------------------
  // Iteration Pattern for RVV Dynamic Vector Length
  // ---------------------------------------------------------------------------

  // 1. Get the total length of the workload.
  %mem = memref.get_global @gv : memref<10xf32>
  %print_mem = memref.cast %mem : memref<10xf32> to memref<*xf32>
  %vl_total = memref.dim %mem, %c0 : memref<10xf32>

  // 2. Set the scale factor, iteration step, and mask.
  %vs = vector.vscale
  %factor = arith.constant 2 : index
  %vl_step = arith.muli %vs, %factor : index
  %mask = arith.constant dense<1> : vector<[2]xi1>
  %vl_total_i32 = index.casts %vl_total : index to i32
  %vl_step_i32 = index.casts %vl_step : index to i32

  // 3. Perform the vectorization.
  %iter_vl = scf.for %i = %c0 to %vl_total step %vl_step
      iter_args(%iter_vl_i32 = %vl_total_i32) -> (i32) {

    %load_vec1 = vector_exp.predication %mask, %iter_vl_i32 : vector<[2]xi1>, i32 {
      %ele = vector.load %mem[%i] : memref<10xf32>, vector<[2]xf32>
      vector.yield %ele : vector<[2]xf32>
    } : vector<[2]xf32>

    %load_vec2 = vector_exp.predication %mask, %iter_vl_i32 : vector<[2]xi1>, i32 {
      %ele = vector.load %mem[%i] : memref<10xf32>, vector<[2]xf32>
      vector.yield %ele : vector<[2]xf32>
    } : vector<[2]xf32>

    %res = "llvm.intr.vp.fadd" (%load_vec1, %load_vec2, %mask, %iter_vl_i32) : 
        (vector<[2]xf32>, vector<[2]xf32>, vector<[2]xi1>, i32) -> vector<[2]xf32>

    vector_exp.predication %mask, %iter_vl_i32 : vector<[2]xi1>, i32 {
      vector.store %res, %mem[%i] : memref<10xf32>, vector<[2]xf32>
      vector.yield
    } : () -> ()

    // Update dynamic vector length.
    %new_vl = arith.subi %iter_vl_i32, %vl_step_i32 : i32
    scf.yield %new_vl : i32
  }

  // CHECK: [0,  2,  4,  6,  8,  10,  12,  14,  8,  9]
  call @printMemrefF32(%print_mem) : (memref<*xf32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
