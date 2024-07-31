module {
  func.func private @rtclock() -> f64
  func.func private @printMemrefF64(memref<*xf64>)
  func.func @main() {
    %alloc = memref.alloc() : memref<3xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : vector<3xf32>
    %cst_0 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : vector<3xf32>
    %0 = timing.start : -> f64
    %1 = arith.addf %cst, %cst_0 : vector<3xf32>
    %2 = timing.end : -> f64
    %3 = arith.subf %0, %2 : f64
    memref.store %3, %alloc[%c0] : memref<3xf64>
    %4 = arith.addi %c0, %c1 : index
    %5 = timing.start : -> f64
    %6 = arith.addf %cst, %cst_0 : vector<3xf32>
    %7 = timing.end : -> f64
    %8 = arith.subf %5, %7 : f64
    memref.store %8, %alloc[%4] : memref<3xf64>
    %9 = arith.addi %4, %c1 : index
    %10 = timing.start : -> f64
    %11 = arith.addf %cst, %cst_0 : vector<3xf32>
    %12 = timing.end : -> f64
    %13 = arith.subf %10, %12 : f64
    memref.store %13, %alloc[%9] : memref<3xf64>
    %14 = arith.addi %9, %c1 : index
    %cast = memref.cast %alloc : memref<3xf64> to memref<*xf64>
    call @printMemrefF64(%cast) : (memref<*xf64>) -> ()
    return
  }
}

