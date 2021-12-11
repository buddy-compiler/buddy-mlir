func @corr_2d(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>)
{
  dip.corr_2d %inputImage, %kernel, %outputImage {centerX = 1, centerY = 1, boundary_option = "constantpadding", constant_value = 0}  : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  return
}
