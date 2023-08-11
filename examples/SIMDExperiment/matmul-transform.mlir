module{
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @matmul(%a : memref<64x64xf32>, %b : memref<64x64xf32>, %c : memref<64x64xf32>) {
    linalg.matmul
      ins(%a, %b: memref<64x64xf32>, memref<64x64xf32>)
      outs(%c: memref<64x64xf32>)
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%module_op: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %module_op : (!pdl.operation) -> !pdl.operation
    %1, %loops:3 = transform.structured.tile %0 [4, 32, 8] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    %2 = get_closest_isolated_parent %1 : (!pdl.operation) -> !pdl.operation
    transform.structured.vectorize %2
    %b = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap}
        %module_op {bufferize_function_boundaries = true}
        : (!pdl.operation) -> !pdl.operation

    %f = transform.structured.match ops{["func.func"]} in %b
      : (!pdl.operation) -> !pdl.operation

    %func = transform.vector.lower_contraction %f
      lowering_strategy = "outerproduct"
        : (!pdl.operation) -> !pdl.operation

    %func_2 = transform.vector.apply_transfer_permutation_patterns %func
        : (!pdl.operation) -> !pdl.operation

    %func_3 = transform.vector.lower_multi_reduction %func_2
      lowering_strategy = "innerparallel"
        : (!pdl.operation) -> !pdl.operation

    %func_4 = transform.vector.split_transfer_full_partial %func_3
      split_transfer_strategy = "linalg-copy"
        : (!pdl.operation) -> !pdl.operation

    %func_5 = transform.vector.transfer_to_scf %func_4
      max_transfer_rank = 1 full_unroll = true
        : (!pdl.operation) -> !pdl.operation

    %func_6 = transform.vector.lower_transfer %func_5
      max_transfer_rank = 1
        : (!pdl.operation) -> !pdl.operation

    %func_7 = transform.vector.lower_shape_cast %func_6
      : (!pdl.operation) -> !pdl.operation

    %func_8 = transform.vector.lower_transpose %func_7
      lowering_strategy = "shuffle_1d"
        : (!pdl.operation) -> !pdl.operation
  }
  func.func @main(){
    // Set up dims.
    %cM = arith.constant 64 : index
    %cN = arith.constant 64 : index
    %cK = arith.constant 64 : index

    // Set Init Value.
    %cf1 = arith.constant 1.0 : f32

    %A = memref.alloc() : memref<64x64xf32>
    %B = memref.alloc() : memref<64x64xf32>
    %C = memref.alloc() : memref<64x64xf32>

    linalg.fill
    ins(%cf1 : f32)
    outs(%A:memref<64x64xf32>)

    linalg.fill
    ins(%cf1 : f32)
    outs(%B:memref<64x64xf32>)

    linalg.fill
    ins(%cf1 : f32)
    outs(%C:memref<64x64xf32>)

    call @matmul(%A, %B, %C) : (memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>) -> ()

    %print_C = memref.cast %C : memref<64x64xf32> to memref<*xf32>
    call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

    memref.dealloc %C : memref<64x64xf32>
    memref.dealloc %B : memref<64x64xf32>
    memref.dealloc %A : memref<64x64xf32>
    return
  }
}
