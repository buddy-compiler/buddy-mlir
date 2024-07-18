func.func @resize_4d_nearest_neighbour_interpolation(%inputImage : memref<?x?x?x?xf32>, %horizontal_scaling_factor : f32, %vertical_scaling_factor : f32, %outputImage : memref<?x?x?x?xf32>) attributes{llvm.emit_c_interface}
{
  dip.resize_4d NEAREST_NEIGHBOUR_INTERPOLATION %inputImage, %horizontal_scaling_factor, %vertical_scaling_factor, %outputImage : memref<?x?x?x?xf32>, f32, f32, memref<?x?x?x?xf32>
  return
}