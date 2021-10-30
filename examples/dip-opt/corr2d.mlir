func @dipCorr2D(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %boundaryOption : index)
{
  dip.Corr2D %inputImage, %kernel, %outputImage, %centerX, %centerY, %boundaryOption : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index
  return
}
