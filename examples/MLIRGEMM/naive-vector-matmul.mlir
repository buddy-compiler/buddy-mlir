#map10 = affine_map<(d0) -> (d0 * 4)> 
module{
  func.func private @printMemrefF64(memref<*xf64>)
  func.func private @rtclock() -> f64

  // Naive matrix multiplication using affine dialect 
  func.func @matmul(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 1 : index
    
    %dim0 = arith.constant 2048: index
    %dim1 = arith.constant 512: index
    %dim2 = arith.constant 2048: index

    affine.for %arg3 = 0 to %dim0 {
      affine.for %arg4 = 0 to %dim1 {
        affine.for %arg5 = 0 to %dim2 {
          %v0 = affine.load %A[%arg3, %arg5] : memref<?x?xf64>
          %v1 = vector.splat %v0 : vector<4xf64> // creates a vector of shape 3x2, by brodcasting to the respective shape
          %idx = affine.apply #map10(%arg4)
          %v2 = affine.vector_load %B[%arg5, %idx] : memref<?x?xf64>, vector<4xf64>
          %v3 = arith.mulf %v1, %v2 : vector<4xf64>
          %v4 = affine.vector_load %C[%arg3, %idx] : memref<?x?xf64>, vector<4xf64>
          %v5 = arith.addf %v3, %v4 : vector<4xf64>
          affine.vector_store %v5, %C[%arg3, %idx] : memref<?x?xf64>, vector<4xf64>
        }
      }
    }
    return
  }
  
func.func @check_result(%A: memref<?x?xf64>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r0 = arith.constant 1 : index

    %dim0 = memref.dim %A, %c0 : memref<?x?xf64>
    %dim1 = memref.dim %A, %c1 : memref<?x?xf64>

    %f1 = arith.constant 2048.0 : f64
    %f2 = arith.constant 0.0 : f64
    %res = memref.alloc(%r0) : memref<?xf64>
    affine.store %f2, %res[%c0] : memref<?xf64>
    
    affine.for %arg3 = 0 to %dim0 {
      affine.for %arg4 = 0 to %dim1 {
        %a = affine.load %A[%arg3, %arg4] : memref<?x?xf64>
        %b = arith.subf %a, %f1  : f64
        %c = arith.mulf %b, %b  : f64
        %r1 = affine.load %res[%c0] : memref<?xf64>
        %r2 = arith.addf %r1, %c  : f64
        affine.store %r2, %res[%c0] : memref<?xf64>
      }
    }

    %print_res = memref.cast %res : memref<?xf64> to memref<*xf64>
    func.call @printMemrefF64(%print_res) : (memref<*xf64>) -> ()
    memref.dealloc %res : memref<?xf64>

    return
  }

  func.func @main(){
    // Set up dims.
    %cM = arith.constant 2048 : index
    %cN = arith.constant 2048 : index
    %cK = arith.constant 2048 : index

    // Set Init Value.
    %cf1 = arith.constant 1.0 : f64
    %cf2 = arith.constant 0.0 : f64
    %A = memref.alloc(%cM, %cK) : memref<?x?xf64>
    %B = memref.alloc(%cK, %cN) : memref<?x?xf64>
    %C = memref.alloc(%cM, %cN) : memref<?x?xf64>


    linalg.fill
     ins(%cf1 : f64)
    outs(%A:memref<?x?xf64>)

    linalg.fill
     ins(%cf1 : f64)
    outs(%B:memref<?x?xf64>)

    linalg.fill
     ins(%cf2 : f64)
    outs(%C:memref<?x?xf64>)


    // Execution times.
    %reps = arith.constant 1 : index

    // Record start time.
    %t_start = call @rtclock() : () -> f64  

    // Execute convolution for specific times.
    affine.for %arg0 = 0 to %reps {
      func.call @matmul(%A, %B, %C) : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()
    }


    // Record end time.
    %t_end = call @rtclock() : () -> f64
    // Get the total running time.
    %t = arith.subf %t_end, %t_start : f64

    func.call @check_result(%C) : (memref<?x?xf64>) -> ()

    // 2 * %cM * %cK * %cN.
    %f1 = arith.constant 2 : index
    %flops1 = arith.muli %cM, %cK : index
    %flops2 = arith.muli %cN, %f1 : index
    %flops3 = arith.muli %flops1, %flops2 : index

    // Calculate FLOPS.
    %num_flops = arith.muli %reps, %flops3 : index
    %num_flops_i = arith.index_cast %num_flops : index to i64
    %num_flops_f = arith.sitofp %num_flops_i : i64 to f64
    %flops = arith.divf %num_flops_f, %t : f64
    // Print the FLOPS.
    vector.print %flops : f64

    memref.dealloc %C : memref<?x?xf64>
    memref.dealloc %B : memref<?x?xf64>
    memref.dealloc %A : memref<?x?xf64>

    return
  }
}
