module attributes {gpu.container_module} {   
  gpu.module @test_module {
    gpu.func @kernel_1(%arg0 : i32) kernel {
      %tIdX = gpu.thread_id x
      %tIdY = gpu.thread_id y
      %tIdZ = gpu.thread_id z

      %bDimX = gpu.block_dim x
      %bDimY = gpu.block_dim y
      %bDimZ = gpu.block_dim z

      %bIdX = gpu.block_id x
      %bIdY = gpu.block_id y
      %bIdZ = gpu.block_id z

      %gDimX = gpu.grid_dim x
      %gDimY = gpu.grid_dim y
      %gDimZ = gpu.grid_dim z

      %laneId = gpu.lane_id

      gpu.printf "thread_id x = %d " %tIdX : index
      gpu.printf "thread_id y = %d " %tIdY : index
      gpu.printf "thread_id z = %d " %tIdZ : index
      gpu.printf "block_dim x = %d " %bDimX : index
      gpu.printf "block_dim y = %d " %bDimY : index
      gpu.printf "block_dim z = %d " %bDimZ : index
      gpu.printf "block_id x = %d " %bIdX : index
      gpu.printf "block_id y = %d " %bIdY : index
      gpu.printf "block_id z = %d " %bIdZ : index
      gpu.printf "grid_dim x = %d " %gDimX : index
      gpu.printf "grid_dim y = %d " %gDimY : index
      gpu.printf "grid_dim z = %d " %gDimZ : index
      gpu.printf "lane_id = %d\n" %laneId : index
      gpu.return
    }
  }
  func.func @main() {
    %cst1 = arith.constant 1 : index 
    %cst2 = arith.constant 1 : index
    %arg0 = arith.constant 1 : i32
    gpu.launch_func @test_module::@kernel_1 blocks in (%cst1, %cst1, %cst1) threads in (%cst2, %cst2, %cst2) args(%arg0 : i32)
    return
  }
}
