module {
  func private @print_memref_f32(memref<*xf32>)
  func @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %arg2, %0[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    return %0 : memref<?x?xf32>
  }

  func @Conv2D(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>,
               %outputImage : memref<?x?xf32>)
  {

    // %outputImage = DIP.Corr2D %inputImage, %kernel : memref<?x?xf32>, memref<?x?xf32> to memref<?x?xf32>

    linalg.conv_2d ins (%inputImage, %kernel: memref<?x?xf32>, memref<?x?xf32>)
                 outs (%outputImage: memref<?x?xf32>)

    // DIP.Corr2D %inputImage, %kernel, %outputImage : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>

    return %inputImage : memref<?x?xf32>
  }

  func @main()
  {
    %cI = constant 2.0 : f32
    %cK = constant 3.0 : f32
    %cO = constant 0.0 : f32

    %inputSize = constant 5 : index
    %kernelSize = constant 3 : index
    %outputSize = constant 3 : index

    %inputImage = call @alloc_2d_filled_f32(%inputSize, %inputSize, %cI) : (index, index, f32) -> memref<?x?xf32>
    %kernel = call @alloc_2d_filled_f32(%kernelSize, %kernelSize, %cK) : (index, index, f32) -> memref<?x?xf32>
    %outputImage = call @alloc_2d_filled_f32(%outputSize, %outputSize, %cO) : (index, index, f32) -> memref<?x?xf32>

    call @Conv2D(%inputImage, %kernel, %outputImage) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
    // %outputImage = call @Conv2D(%inputImage, %kernel) : (memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>

    %printInputImage = memref.cast %inputImage : memref<?x?xf32> to memref<*xf32>
    // call @print_memref_f32(%printInputImage) : memref<*xf32> -> ()

    return
  }
}
