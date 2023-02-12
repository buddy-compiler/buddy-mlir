#map10 = affine_map<(d0) -> (d0 * 64)> 
#map11 = affine_map<(d0) -> (d0 * 64 + 64)>  
#map12 = affine_map<(d0) -> (d0 * 256)>
#map13 = affine_map<(d0) -> (d0 * 256 + 256)>  
#map14 = affine_map<(d0) -> (d0 * 8)> 
#map15 = affine_map<(d0) -> (d0 * 8 + 8)> 
#map16 = affine_map<(d0) -> (d0 * 16)> 
#map17 = affine_map<(d0) -> (d0 * 16 + 16)> 

module{
  // print result
  func.func private @printMemrefF64(memref<*xf64>)
  // get time
  func.func private @rtclock() -> f64
  // check matmultiplicaion
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


  // naive matrix multiplication using affine dialect and tiling and packing and unrolling
  func.func @matmul(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 64 : index
    %c3 = arith.constant 256 : index
    %c4 = arith.constant 8 : index
    
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 32 {
        %0 = memref.alloc(%c2, %c3) : memref<?x?xf64>
        affine.for %arg5 = #map10(%arg4) to #map11(%arg4) {
          affine.for %arg6 = #map12(%arg3) to #map13(%arg3) {
            %1 = affine.load %A[%arg5, %arg6] : memref<?x?xf64>
            affine.store %1, %0[%arg4 * -64 + %arg5, %arg3 * -256 + %arg6] : memref<?x?xf64>
          }
        }
        affine.for %arg5 = 0 to 256 {
          //256*8
          %1 = memref.alloc(%c3, %c4) : memref<?x?xf64>
          affine.for %arg6 = #map12(%arg3) to #map13(%arg3) {
            affine.for %arg7 = #map14(%arg5) to #map15(%arg5) {
              %2 = affine.load %B[%arg6, %arg7] : memref<?x?xf64>
              affine.store %2, %1[%arg3 * -256 + %arg6, %arg5 * -8 + %arg7] : memref<?x?xf64>
            }
          }

          affine.for %arg6 = #map16(%arg4) to #map17(%arg4){
            affine.for %arg7 = 0 to 256 {
              affine.for %arg8 = 0 to 8 {
                %3 = affine.load %1[%arg7, %arg8] : memref<?x?xf64>
                // 1
                %2 = affine.load %0[%arg4 * -64 + %arg6 * 4 + 0, %arg7] : memref<?x?xf64>
                %4 = affine.load %C[%arg6 * 4 + 0, %arg5 * 8 + %arg8] : memref<?x?xf64>
                %5 =  arith.mulf %2, %3 : f64
                %6 = arith.addf %4, %5 : f64
                affine.store %6, %C[%arg6 * 4 + 0, %arg5 * 8 + %arg8] : memref<?x?xf64>
                // 2
                %7 = affine.load %0[%arg4 * -64 + %arg6 * 4 + 1, %arg7] : memref<?x?xf64>
                %8 = affine.load %C[%arg6 * 4 + 1, %arg5 * 8 + %arg8] : memref<?x?xf64>
                %9 =  arith.mulf %3, %7 : f64
                %10 = arith.addf %8, %9 : f64
                affine.store %10, %C[%arg6 * 4 + 1, %arg5 * 8 + %arg8] : memref<?x?xf64>
                // 3
                %11 = affine.load %0[%arg4 * -64 + %arg6 * 4 + 2, %arg7] : memref<?x?xf64>
                %12 = affine.load %C[%arg6 * 4 + 2, %arg5 * 8 + %arg8] : memref<?x?xf64>
                %13 =  arith.mulf %3, %11 : f64
                %14 = arith.addf %12, %13 : f64
                affine.store %14, %C[%arg6 * 4 + 2, %arg5 * 8 + %arg8] : memref<?x?xf64>
                // 4
                %15 = affine.load %0[%arg4 * -64 + %arg6 * 4 + 3, %arg7] : memref<?x?xf64>
                %16 = affine.load %C[%arg6 * 4 + 3, %arg5 * 8 + %arg8] : memref<?x?xf64>
                %17 =  arith.mulf %3, %15 : f64
                %18 = arith.addf %15, %16 : f64
                affine.store %14, %C[%arg6 * 4 + 3, %arg5 * 8 + %arg8] : memref<?x?xf64>
              }
            }
          }
          memref.dealloc %1 : memref<?x?xf64>
        }
        memref.dealloc %0 : memref<?x?xf64>
      }
    }
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