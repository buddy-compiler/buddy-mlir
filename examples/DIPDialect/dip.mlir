func.func @corr_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32)
{
  dip.corr_2d CONSTANT_PADDING %inputImage, %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @corr_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32)
{
  dip.corr_2d REPLICATE_PADDING %inputImage, %kernel, %outputImage, %centerX, %centerY , %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @rotate_2d(%inputImage : memref<?x?xf32>, %angle : f32, %outputImage : memref<?x?xf32>)
{
  dip.rotate_2d %inputImage, %angle, %outputImage : memref<?x?xf32>, f32, memref<?x?xf32>
  return
}
