func @main() {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index

  %A = memref.alloc() : memref<8x8xf32>
  %U = memref.cast %A :  memref<8x8xf32> to memref<*xf32>

  // Initialize memref with zeros because we do load and store to in every test
  // to verify that we process each element of the iteration space once.
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    memref.store %c0, %A[%i, %j] : memref<8x8xf32>
  }

  // 1. (%i, %i) = (0, 8) to (8, 8) step (1, 1)
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    %0 = arith.muli %i, %c8 : index
    %1 = arith.addi %j, %0  : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = memref.load %A[%i, %j] : memref<8x8xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %A[%i, %j] : memref<8x8xf32>
  }

  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    memref.store %c0, %A[%i, %j] : memref<8x8xf32>
  }

  // 2. (%i, %i) = (0, 8) to (8, 8) step (2, 1)
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c2, %c1) {
    %0 = arith.muli %i, %c8 : index
    %1 = arith.addi %j, %0  : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = memref.load %A[%i, %j] : memref<8x8xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %A[%i, %j] : memref<8x8xf32>
  }

  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    memref.store %c0, %A[%i, %j] : memref<8x8xf32>
  }

  // 3. (%i, %i) = (0, 8) to (8, 8) step (1, 2)
  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c2) {
    %0 = arith.muli %i, %c8 : index
    %1 = arith.addi %j, %0  : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = memref.load %A[%i, %j] : memref<8x8xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %A[%i, %j] : memref<8x8xf32>
  }

  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  memref.dealloc %A : memref<8x8xf32>

  return
}

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
