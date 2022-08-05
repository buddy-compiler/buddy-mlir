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

func.func @resize_2d_nearest_neighbour_interpolation(%inputImage : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?xf32>)
{
  dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}

func.func @resize_2d_bilinear_interpolation(%inputImage : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?xf32>)
{
  dip.resize_2d BILINEAR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}


func.func @erosion_2d_constant_padding_flat(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue: f32)
{
  dip.erosion_2d CONSTANT_PADDING %inputImage, FLAT %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @erosion_2d_replicate_padding_flat(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32)
{
  dip.erosion_2d REPLICATE_PADDING %inputImage, FLAT %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return 
}
 
func.func @erosion_2d_constant_padding_non_flat(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32) 
{
  dip.erosion_2d CONSTANT_PADDING %inputImage, NONFLAT %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return 
}

func.func @erosion_2d_replicate_padding_non_flat(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32) 
{
  dip.erosion_2d REPLICATE_PADDING %inputImage, NONFLAT %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return 
}

func.func @dilation_2d_constant_padding_flat(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue: f32)
{
  dip.dilation_2d CONSTANT_PADDING %inputImage, FLAT %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @dilation_2d_replicate_padding_flat(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32)
{
  dip.dilation_2d REPLICATE_PADDING %inputImage, FLAT %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return 
}
 
func.func @dilation_2d_constant_padding_non_flat(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32) 
{
  dip.dilation_2d CONSTANT_PADDING %inputImage, NONFLAT %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return 
}

func.func @dilation_2d_replicate_padding_non_flat(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32) 
{
  dip.dilation_2d REPLICATE_PADDING %inputImage, NONFLAT %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return 
}

