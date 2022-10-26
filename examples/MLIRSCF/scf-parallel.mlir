func.func @main() {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  %lb = arith.constant 0 : index
  %ub = arith.constant 40000 : index

  %A = memref.alloc() : memref<40000x40000xf32>
  %U = memref.cast %A :  memref<40000x40000xf32> to memref<*xf32>

  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    memref.store %c0, %A[%i, %j] : memref<40000x40000xf32>
  }

  scf.parallel (%i, %j) = (%lb, %lb) to (%ub, %ub) step (%c1, %c1) {
    %0 = arith.muli %i, %c8 : index
    %1 = arith.addi %j, %0  : index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = memref.load %A[%i, %j] : memref<40000x40000xf32>
    %5 = arith.addf %3, %4 : f32
    memref.store %5, %A[%i, %j] : memref<40000x40000xf32>
  }

  memref.dealloc %A : memref<40000x40000xf32>

  return
}

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
