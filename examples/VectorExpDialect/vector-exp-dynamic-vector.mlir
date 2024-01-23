#map = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>

func.func private @printMemrefI32(memref<*xi32>)

func.func @alloc_mem_i32(%init: i32) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %mem = memref.alloc(%c20) : memref<?xi32>
  scf.for %idx0 = %c0 to %c20 step %c1 {
    memref.store %init, %mem[%idx0] : memref<?xi32>
  }
  return %mem : memref<?xi32>
}

func.func @vector_add(%input1: memref<?xi32>, %input2: memref<?xi32>, %output: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  // Get the dimension of the workload.
  %dim_size = memref.dim %input1, %c0 : memref<?xi32>
  // Perform dynamic vector addition.
  // Returns four times the physical vl for element type i32.
  %vl = vector_exp.get_vl i32, 4 : index

  scf.for %idx = %c0 to %dim_size step %vl { // Tiling
    %it_vl = affine.min #map(%idx)[%vl, %dim_size]
    vector_exp.set_vl %it_vl : index {
      %vec_input1 = vector.load %input1[%idx] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      %vec_input2 = vector.load %input2[%idx] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      %vec_output = arith.addi %vec_input1, %vec_input2 : vector<[1]xi32> // vector<?xi32>
      vector.store %vec_output, %output[%idx] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      vector.yield
    }
  }
  return
}

func.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  %input_mem = call @alloc_mem_i32(%c1_i32) : (i32) -> memref<?xi32>
  %result_mem = call @alloc_mem_i32(%c0_i32) : (i32) -> memref<?xi32>

  call @vector_add(%input_mem, %input_mem, %result_mem) : (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()

  %print_result_mem = memref.cast %result_mem : memref<?xi32> to memref<*xi32>
  call @printMemrefI32(%print_result_mem) : (memref<*xi32>) -> ()

  %ret = arith.constant 0 : i32
  return %ret : i32
}
