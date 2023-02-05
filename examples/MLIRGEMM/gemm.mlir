#map0 = affine_map<(d0) -> (d0+10)>
  

#map10 = affine_map<(d0) -> (d0 * 64)> 
#map11 = affine_map<(d0) -> (d0 * 64 + 64)>  
#map12 = affine_map<(d0) -> (d0 * 256)>
#map13 = affine_map<(d0) -> (d0 * 256 + 256)>  
#map14 = affine_map<(d0) -> (d0 * 8)> 
#map15 = affine_map<(d0) -> (d0 * 8 + 8)> 
#map16 = affine_map<(d0) -> (d0 * 16)> 
#map17 = affine_map<(d0) -> (d0 * 16 + 16)> 

module{

    func.func private @printMemrefF64(memref<*xf64>)
    func.func @matmul(%a : memref<?x?xf64>, %b : memref<?x?xf64>, %c : memref<?x?xf64>) {
      linalg.matmul 
        ins(%a, %b: memref<?x?xf64>, memref<?x?xf64>)
       outs(%c:memref<?x?xf64>)
      return
    }


    func.func @matmul1(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {
      affine.for %arg3 = 0 to 2048 {
          affine.for %arg4 = 0 to 2048 {
              affine.for %arg5 = 0 to 2048 {
                  %a = affine.load %A[%arg3, %arg5] : memref<?x?xf64>
                  %b = affine.load %B[%arg5, %arg4] : memref<?x?xf64>
                  %ci = affine.load %C[%arg3, %arg4] : memref<?x?xf64>
                  %p = arith.mulf %a, %b  : f64
                  %co = arith.addf %ci, %p : f64
                  affine.store %co, %C[%arg3, %arg4] : memref<?x?xf64>
              }
          }
      }
      return
    }


    func.func @matmul2(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 1 : index
      %dim0 = memref.dim %A, %c0 : memref<?x?xf64>
      %dim1 = memref.dim %A, %c1 : memref<?x?xf64>
      %dim2 = memref.dim %B, %c2 : memref<?x?xf64>

      vector.print %dim0 : index
      vector.print %dim1 : index
      vector.print %dim2 : index
 
      affine.for %arg3 = 0 to %dim0 {
        affine.for %arg4 = 0 to %dim1 {
          affine.for %arg5 = 0 to %dim2 {
            %a = affine.load %A[%arg3, %arg5] : memref<?x?xf64>
            %b = affine.load %B[%arg5, %arg4] : memref<?x?xf64>
            %ci = affine.load %C[%arg3, %arg4] : memref<?x?xf64>
            %p = arith.mulf %a, %b  : f64
            %co = arith.addf %ci, %p : f64
            affine.store %co, %C[%arg3, %arg4] : memref<?x?xf64>
          }
        }
      }
      return
    }

    func.func @matmul3(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      affine.for %arg3 = 0 to 8 {
        affine.for %arg4 = 0 to 33 {
          affine.for %arg5 = 0 to 2048 {
            affine.for %arg6 = #map10(%arg4) to #map11(%arg4) {
              %0 = memref.alloc(%c1) : memref<?xf64>
              %1 = affine.load %C[%arg6, %arg5] : memref<?x?xf64>
              affine.store %1, %0[%c0] : memref<?xf64>
              affine.for %arg7 = 0 to 256 {
                %3 = affine.load %A[%arg6, %arg3 * 256 + %arg7] : memref<?x?xf64>
                %4 = affine.load %B[%arg3 * 256 + %arg7, %arg5] : memref<?x?xf64>
                %5 = affine.load %0[0] : memref<?xf64>
                %6 =  arith.mulf %3, %4 : f64
                %7 = arith.addf %5, %6 : f64
                affine.store %7, %0[0] : memref<?xf64>
              }
              %7 = affine.load %0[%c0] : memref<?xf64>
              affine.store %7, %C[%arg6, %arg5] : memref<?x?xf64>
            }
          }
        }
      }
      return
    }

    func.func @matmul4(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {
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
                    affine.for %arg9 = 0 to 4 {
                      %2 = affine.load %0[%arg4 * -64 + %arg6 * 4 + %arg9, %arg7] : memref<?x?xf64>
                      %3 = affine.load %1[%arg7, %arg8] : memref<?x?xf64>
                      %4 = affine.load %C[%arg6 * 4 + %arg9, %arg5 * 8 + %arg8] : memref<?x?xf64>
                      %5 =  arith.mulf %2, %3 : f64
                      %6 = arith.addf %4, %5 : f64
                      affine.store %6, %C[%arg6 * 4 + %arg9, %arg5 * 8 + %arg8] : memref<?x?xf64>
                    }
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
   
    func.func @matmul5(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>) {
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
                    %17 =  arith.mulf %2, %15 : f64
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
       %C1 = memref.alloc(%cM, %cN) : memref<?x?xf64>
       %C2 = memref.alloc(%cM, %cN) : memref<?x?xf64>


       linalg.fill
        ins(%cf1 : f64)
       outs(%A:memref<?x?xf64>)

       linalg.fill
        ins(%cf1 : f64)
       outs(%B:memref<?x?xf64>)

       linalg.fill
        ins(%cf2 : f64)
       outs(%C:memref<?x?xf64>)

       linalg.fill
        ins(%cf2 : f64)
       outs(%C1:memref<?x?xf64>)

       linalg.fill
        ins(%cf2 : f64)
       outs(%C2:memref<?x?xf64>)

       %print_A = memref.cast %A : memref<?x?xf64> to memref<*xf64>
       //call @printMemreff64(%print_A) : (memref<*xf64>) -> ()

       %print_B = memref.cast %B : memref<?x?xf64> to memref<*xf64>
       //call @printMemreff64(%print_B) : (memref<*xf64>) -> ()

       //call @matmul1(%A, %B, %C1) : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()

       //call @matmul2(%A, %B, %C2) : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()


       // call @matmul3(%A, %B, %C) : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()

       // call @matmul4(%A, %B, %C) : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()

       call @matmul5(%A, %B, %C) : (memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) -> ()


       // Print output.
       %print_C = memref.cast %C : memref<?x?xf64> to memref<*xf64>
       //call @printMemrefF64(%print_C) : (memref<*xf64>) -> ()

       memref.dealloc %C : memref<?x?xf64>
       memref.dealloc %C1 : memref<?x?xf64>
       memref.dealloc %C2 : memref<?x?xf64>
       memref.dealloc %B : memref<?x?xf64>
       memref.dealloc %A : memref<?x?xf64>  
       return 
    }
}



 




