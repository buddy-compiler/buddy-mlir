module attributes {transform.with_named_sequence} {
  transform.named_sequence @codegen(%arg0: !transform.any_op) {
    %conv2d = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    %padded, %pad, %copy = transform.structured.pad %conv2d
      { 
        copy_back_op = "none", 
        pack_paddings = [0, 0, 1, 1, 0, 0, 1, 1],
        pad_to_multiple_of = [1, 1, 16, 16],
        padding_dimensions = [0, 1, 2, 3],
        padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32, 0.0 : f32]
      } 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %tiled_conv2d, %forall_op = transform.structured.tile_using_forall %padded
      tile_sizes [1, 64] (mapping = [#gpu.block<y>, #gpu.block<x>])
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_conv2d_op, %loops = transform.structured.tile_using_forall %tiled_conv2d
      tile_sizes [16, 16] (mapping = [#gpu.thread<y>, #gpu.thread<x>])
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Perform canonicalization.
    %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %1 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %1 : !transform.any_op
    %all_loops = transform.structured.match interface{LoopLikeInterface}
        in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op
    transform.apply_patterns to %1 {
    transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op

    %tiled_conv2d_prod, %trasformed = transform.structured.convert_conv2d_to_img2col %tiled_conv2d_op : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_img2col, %forall_op_0 = transform.structured.tile_using_forall %tiled_conv2d_prod num_threads [1, 4](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %generic_matmul = transform.get_producer_of_operand %trasformed[0] : (!transform.any_op) -> !transform.any_op

    %tiled_matmul, %forall_op_1 = transform.structured.tile_using_forall %generic_matmul num_threads [1, 16](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %2 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %2 : !transform.any_op
    %all_loops_2 = transform.structured.match interface{LoopLikeInterface}
        in %2
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops_2 : !transform.any_op
    transform.apply_patterns to %2 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.any_op


    %3 = transform.structured.vectorize_children_and_apply_patterns %2 : (!transform.any_op) -> !transform.any_op

    // Perform canonicalization.
    transform.apply_patterns to %3 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %3 : !transform.any_op
    %all_loops_3 = transform.structured.match interface{LoopLikeInterface}
        in %3
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops_3 : !transform.any_op
    transform.apply_patterns to %3 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.any_op    

    // Match bufferization.alloc_tensors inside the forall op
    %scf_forall = transform.structured.match ops{["scf.forall"]} attributes{mapping = [#gpu.block<y>, #gpu.block<x>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %alloc_tensor_ops = transform.structured.match ops{["bufferization.alloc_tensor"]} in %scf_forall : (!transform.any_op) -> !transform.any_op

    // Bufferize the alloc_tensor ops to memref.alloc ops.
    // The memory_space attribute for GPU Dialect 0 means global memory, 3 means workgroup memory address, 5 means private memory address.
    // According to https://discourse.llvm.org/t/rfc-memref-memory-shape-as-attribute/2229
    %buffer, %new_ops = transform.structured.bufferize_to_allocation %alloc_tensor_ops {memory_space = 3 } : !transform.any_op

    // Eliminate empty tensors and erase unnecessary inputs.
    transform.structured.eliminate_empty_tensors %arg0 : !transform.any_op
    %func_eras = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %func_eras {
        transform.apply_patterns.linalg.erase_unnecessary_inputs
      } : !transform.any_op

    // Bufferize the remaining operations in one time.
    %11 = transform.bufferization.one_shot_bufferize %arg0 { bufferize_function_boundaries = true, function_boundary_type_conversion = 1 : i32} : (!transform.any_op) -> !transform.any_op

    // Erase dead alloc and stores.
    %12 = transform.structured.match ops{["func.func"]} in %11 : (!transform.any_op) -> !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %12 : (!transform.any_op) -> ()

    // Generate GPU launch.
    %13 = transform.structured.match ops{["func.func"]} in %11 : (!transform.any_op) -> !transform.any_op
    %gpu_launch = transform.gpu.map_forall_to_blocks %13 { generate_gpu_launch } : (!transform.any_op) -> !transform.any_op

    // Rewrite bufferized scf.forall ops to distributed gpu.thread_id attribute.
    %mapped = transform.gpu.map_nested_forall_to_threads %gpu_launch block_dims = [64, 2, 1] warp_size = 32 : (!transform.any_op) -> !transform.any_op

    %15 = transform.structured.match ops{["func.func"]} in %11 : (!transform.any_op) -> !transform.any_op

    // Removes unnecessary GPU barriers from the function.
    // %15 = transform.buddy.eliminate_gpu_barriers %14 : (!transform.any_op) -> !transform.any_op

    // Perform canonicalization.
    transform.apply_patterns to %15 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %15 : !transform.any_op
    %all_loops_4 = transform.structured.match interface{LoopLikeInterface}
        in %15
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops_4 : !transform.any_op
    transform.apply_patterns to %15 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.any_op

    // Identify static memory allocations within the given region,
    // and move them to a higher level (hoisting).
    transform.buddy.hoist_static_alloc %15 : (!transform.any_op) -> ()

    // Collects patterns for folding memref aliasing ops (memref.subview) into consumer load/store ops (affine.load, memref.load, nvgpu.ldmatrix, vector.load, vector.transfer_read, affine.store, memref.store, etc.) and other ops (e.g., memref.subview).
    transform.apply_patterns to %15 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    // Collects patterns for extracting address computations from operations with memory accesses such that these memory accesses use only a base pointer.
    transform.apply_patterns to %15 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    // Perform canonicalization.
    transform.apply_patterns to %15 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %15 : !transform.any_op
    %all_loops_5 = transform.structured.match interface{LoopLikeInterface}
        in %15
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops_5 : !transform.any_op
    transform.apply_patterns to %15 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.any_op

    // Adds patterns that unroll vectors to a native tile size for GPUs with mma operations
    transform.apply_patterns to %15 {
      transform.apply_patterns.buddy.unroll_vectors_gpu_mma_sync
    } : !transform.any_op

    // Insert a gpu.barrier after a given scf.for loop
    %16 = transform.structured.match ops{["scf.for"]} in %15 : (!transform.any_op) -> !transform.op<"scf.for">
    // transform.buddy.synchronize_loop %16 : (!transform.op<"scf.for">) -> ()


    transform.apply_patterns to %15 {
                transform.apply_patterns.memref.fold_memref_alias_ops
              } : !transform.any_op
    transform.apply_cse to %15 : !transform.any_op

    // Hoist vector.transfer_read / vector.transfer_write pairs out of immediately enclosing scf::ForOp iteratively
    // Warning: Deprecated
    %17 = transform.structured.hoist_redundant_vector_transfers %15 : (!transform.any_op) -> !transform.any_op

    // Perform canonicalization.
    transform.apply_patterns to %17 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %17 : !transform.any_op
    %all_loops_6 = transform.structured.match interface{LoopLikeInterface}
        in %17
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops_6 : !transform.any_op
    transform.apply_patterns to %17 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.any_op

    // This converts slices of operations containing vector.contract op into
    // mma operations, targetting warp level tensorcore operations.
    transform.buddy.vector.vector_to_mma_conversion %17 {use_mma_sync} : (!transform.any_op) -> ()

    // %18 = transform.buddy.eliminate_gpu_barriers %17 : (!transform.any_op) -> !transform.any_op

    // Perform canonicalization.
    transform.apply_patterns to %17 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %17 : !transform.any_op
    %all_loops_7 = transform.structured.match interface{LoopLikeInterface}
        in %17
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops_7 : !transform.any_op
    transform.apply_patterns to %17 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.any_op

    %19 = transform.structured.match ops{["gpu.launch"]} in %17 : (!transform.any_op) -> !transform.any_op
    %fwfa = transform.structured.match ops{["memref.alloc"]} in %19 : (!transform.any_op) -> !transform.op<"memref.alloc">

    // Do multi-buffering/array expansion to remove dependencies on the temporary allocation between consecutive loop iterations.
    transform.memref.multibuffer %fwfa {factor = 3 : i64, skip_analysis} : (!transform.op<"memref.alloc">) -> !transform.any_op

    transform.apply_patterns to %17 {
      transform.apply_patterns.vector.transfer_to_scf full_unroll = true
    } : !transform.any_op
    transform.apply_patterns to %17 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %17 : !transform.any_op
    %all_loops_8 = transform.structured.match interface{LoopLikeInterface}
        in %17
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops_8 : !transform.any_op
    transform.apply_patterns to %17 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.any_op

    // Convert sync copies to shared memory to async.
    // transform.buddy.create_async_groups %17 {use_mma_sync} : (!transform.any_op) -> ()
    transform.apply_patterns to %17 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    %all_loops_9 = transform.structured.match interface{LoopLikeInterface}
        in %17
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops_9 : !transform.any_op
    transform.apply_cse to %17 : !transform.any_op


    %20 = transform.structured.match ops{["nvgpu.mma.sync"]} in %17 : (!transform.any_op) -> !transform.any_op
    %21 = transform.get_parent_op %20 {deduplicate, op_name = "scf.for"} : (!transform.any_op) -> !transform.any_op
    // This applies software pipelining to a given scf.for loop.
    // The pipelining strategy will look for a copy to shared memory and pipeline it to overlap it with the rest of the loop.
    // %22 = transform.buddy.pipeline_shared_memory_copies %21 {depth = 3 : i64, use_mma_sync, peel_epilogue} : (!transform.any_op) -> !transform.any_op

    // Perform canonicalization.
    transform.apply_patterns to %17 {
      transform.apply_patterns.vector.lower_masks
    } : !transform.any_op
    transform.apply_patterns to %17 {
      transform.apply_patterns.vector.materialize_masks
    } : !transform.any_op
    transform.apply_patterns to %17 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op

    %all_loops_10 = transform.structured.match interface{LoopLikeInterface}
        in %17
        : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops_10 : !transform.any_op
    transform.apply_cse to %17 : !transform.any_op

    transform.yield
  }
} // module
