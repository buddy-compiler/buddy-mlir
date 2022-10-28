func.func @corr_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32)
{
  dip.corr_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %centerX, %centerY, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
  return
}

func.func @corr_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %centerX : index, %centerY : index, %constantValue : f32)
{
  dip.corr_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %centerX, %centerY , %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, f32
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

func.func @erosion_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32)
{
  dip.erosion_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %copymemref, %centerX, %centerY, %iterations, %constantValue: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @erosion_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32)
{
  dip.erosion_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %copymemref, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @dilation_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32)
{
  dip.dilation_2d <CONSTANT_PADDING> %inputImage,  %kernel, %outputImage, %copymemref, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @dilation_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32)
{
  dip.dilation_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %copymemref, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @opening_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32)
{
  dip.opening_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @opening_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32)
{
  dip.opening_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @closing_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32)
{
  dip.closing_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @closing_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32)
{
  dip.closing_2d <REPLICATE_PADDING> %inputImage,  %kernel, %outputImage, %outputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>,memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @tophat_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>,%outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32)
{
  dip.tophat_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @tophat_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32)
{
  dip.tophat_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @bottomhat_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue: f32)
{
  dip.bottomhat_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @bottomhat_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>, %outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32)
{
  dip.bottomhat_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}

func.func @morphgrad_2d_constant_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>,%outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index,%iterations : index, %constantValue: f32)
{
  dip.morphgrad_2d <CONSTANT_PADDING> %inputImage, %kernel, %outputImage, %outputImage1, %outputImage2,  %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index,index, f32
  return
}

func.func @morphgrad_2d_replicate_padding(%inputImage : memref<?x?xf32>, %kernel : memref<?x?xf32>, %outputImage : memref<?x?xf32>, %outputImage1 : memref<?x?xf32>,%outputImage2 : memref<?x?xf32>, %inputImage1 : memref<?x?xf32>, %copymemref : memref<?x?xf32>, %copymemref1 : memref<?x?xf32>, %centerX : index, %centerY : index, %iterations : index, %constantValue : f32)
{
  dip.morphgrad_2d <REPLICATE_PADDING> %inputImage, %kernel, %outputImage, %outputImage1,%outputImage2, %inputImage1, %copymemref, %copymemref1, %centerX, %centerY, %iterations, %constantValue : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32
  return
}
