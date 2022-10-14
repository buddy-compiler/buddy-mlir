func.func @main() {
  %c1 = arith.constant 1 : index 
  %c10 = arith.constant 10 : index 
  %c100 = arith.constant 100 : index 
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %cst2 = arith.constant 2 : i32
  %mem = memref.alloc() : memref<100xi32>
  %mem_cast = memref.cast %mem : memref<100xi32> to memref<*xi32>
  gpu.host_register %mem_cast : memref<*xi32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1) 
             threads(%tx, %ty, %tz) in (%block_x = %c100, %block_y = %c1, %block_z = %c1) {
      memref.store %cst0, %mem[%tx] : memref<100xi32>
      gpu.terminator
  }
  call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()  
  %t0, %r0 = async.execute() -> !async.value<memref<100xi32>> {
    %b0 = gpu.alloc() : memref<100xi32>
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)   
               threads(%tx, %ty, %tz) in (%block_x = %c100, %block_y = %c1, %block_z = %c1) {
      memref.store %cst1, %b0[%tx] : memref<100xi32>
      gpu.terminator
    }
    async.yield %b0 : memref<100xi32>
  }
  %t1, %r1 = async.execute() -> !async.value<memref<100xi32>> {
    %b0 = gpu.alloc() : memref<100xi32>
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1) 
               threads(%tx, %ty, %tz) in (%block_x = %c100, %block_y = %c1, %block_z = %c1) {
      memref.store %cst2, %b0[%tx] : memref<100xi32>
      gpu.terminator
    }
    async.yield %b0 : memref<100xi32>
  }
  %t2 = async.execute [%t0, %t1] (%r0 as %mem0 : !async.value<memref<100xi32>>, 
                        %r1 as %mem1 : !async.value<memref<100xi32>>) {
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1) 
               threads(%tx, %ty, %tz) in (%block_x = %c100, %block_y = %c1, %block_z = %c1) {
      %lhs = memref.load %mem0[%tx] : memref<100xi32>
      %rhs = memref.load %mem1[%tx] : memref<100xi32>
      %tmp = arith.addi %lhs, %rhs : i32
      memref.store %tmp, %mem[%tx] : memref<100xi32>
      gpu.terminator
    }
    gpu.dealloc %mem0 : memref<100xi32>
    gpu.dealloc %mem1 : memref<100xi32>
    async.yield
  }
  async.await %t2 : !async.token
  call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()
  return 
}
func.func private @printMemrefI32(%ptr : memref<*xi32>)

