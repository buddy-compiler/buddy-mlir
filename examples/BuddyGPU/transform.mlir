module attributes { transform.with_named_sequence } {
  transform.named_sequence @codegen(%arg0: !transform.any_op) {
    // Match the target operations and assign them to SSA values.
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0
      : (!transform.any_op) -> !transform.any_op

    // Perform tiling for the grid.
    // For the matrix multiplication of 5376x2048 and 2048x5376, the compilation
    // strategy sets the tile size for grid-based partitioning to 128x256.
    // This means that each 128x256 matmul tile is computed within a GPU block,
    // while multiple such blocks are computed in parallel across the grid.
    // `tile_sizes` specify the dimensions of the tiled matmul result.
    // `%tiled_op` is the tiled matmul operation within the `scf.forall` loop.
    // `%forall_op` is the `scf.forall` loop that maintains tile information.
    %tiled_op, %forall_op = transform.structured.tile_using_forall %matmul
      tile_sizes [128, 256] (mapping = [#gpu.block<y>, #gpu.block<x>])
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
} // module
