func.func @resize_2d_nearest_neighbour_interpolation(%inputImage : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.resize_2d NEAREST_NEIGHBOUR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}

func.func @resize_2d_bilinear_interpolation(%inputImage : memref<?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.resize_2d BILINEAR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?xf32>, f32, f32, memref<?x?xf32>
  return
}