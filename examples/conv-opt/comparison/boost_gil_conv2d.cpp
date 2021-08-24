#include <boost/gil/extension/io/png.hpp>
#include <boost/gil.hpp>
#include "../kernels.h"
#include <time.h>
#include <iostream>

namespace gil = boost::gil;

int main(int argc, char* argv[])
{
  // Declare input image
  gil::gray8_image_t image; 
    
  // Read input image
  gil::read_image(argv[1], image, gil::png_tag{});

  // Declare output image
  gil::gray8_image_t output(image.dimensions());
    
  // Create a 2D GIL kernel
  gil::detail::kernel_2d<float> kernel(sobel3x3KernelAlign, 9, 1, 1);

  clock_t start,end;
  start = clock();
  gil::detail::convolve_2d(gil::view(image), kernel, gil::view(output));
  end = clock();
  std::cout << "Execution time: " 
       << (double)(end - start) / CLOCKS_PER_SEC << " s" << std::endl;

  // Save obtained image
  gil::write_view(argv[2], gil::view(output), gil::png_tag{});
}
