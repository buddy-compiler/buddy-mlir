//===- ImgContainer.h -----------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Image container descriptor (without OpenCV dependency).
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DIP_IMGCONTAINER
#define FRONTEND_INTERFACES_BUDDY_DIP_IMGCONTAINER

#include "buddy/Core/Container.h"
#include <cstring>
#include <fstream>
#include <memory>
#include <png.h>

namespace dip {
enum ImageModes {
  DIP_GRAYSCALE = 0,
  DIP_RGB = 1,
};

inline bool isBigEndian() {
  int num = 1;
  char *ptr = (char *)&num;
  return (*ptr == 0);
}

template <typename T, size_t N> class Image : public MemRef<T, N> {
public:
  // Constructor initializes the image by loading from a file.
  // Params:
  //   filename: Specifies the path to the image file.
  //   mode: Specifies the image mode (e.g., DIP_GRAYSCALE, DIP_RGB).
  //   norm: Indicates whether to normalize pixel values (default is false).
  Image(std::string filename, ImageModes mode, bool norm = false);

  // Retrieves the name of the current image format as a string.
  std::string getFormatName() const {
    switch (this->imageFormat) {
    case ImageFormat::BMP:
      return "BMP";
    case ImageFormat::PNG:
      return "PNG";
    default:
      return "Unsupported format";
    }
  }
  // Returns the width of the image in pixels.
  size_t getWidth() const { return this->width; }
  // Returns the height of the image in pixels.
  size_t getHeight() const { return this->height; }
  // Returns the bit depth of the image.
  int getBitDepth() const { return this->bitDepth; }

private:
  // Enum to represent supported image formats.
  enum class ImageFormat {
    ERROR, // Represents an error or unsupported format.
    BMP,   // BMP file format.
    PNG,   // PNG file format.
  } imageFormat;
  // Mode of the image (e.g., DIP_GRAYSCALE, DIP_RGB).
  ImageModes imageMode;
  // Width of the image in pixels.
  size_t width;
  // Height of the image in pixels.
  size_t height;
  // Bit depth of the image.
  int bitDepth;
  // Normalization flag.
  bool isNorm;
  // Determines the image format from raw file data.
  void determineFormat(const std::vector<uint8_t> &fileData);
  // Decodes a BMP image from raw file data.
  bool decodeBMP(const std::vector<uint8_t> &fileData);
  // Decodes a PNG image from raw file data.
  bool decodePNG(const std::vector<uint8_t> &fileData);
};

// Image Container Constructor
// Constructs an image container object from the image file path.
template <typename T, std::size_t N>
Image<T, N>::Image(std::string filePath, ImageModes mode, bool norm)
    : imageMode(mode), isNorm(norm) {
  // ---------------------------------------------------------------------------
  // 1. Read the image file into a std::vector.
  // ---------------------------------------------------------------------------
  // Open the file in binary mode and position the file pointer at the end of
  // the file.
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  // Check if the file was successfully opened.
  if (!file) {
    throw std::runtime_error("Error: Unable to open file at " + filePath);
  }
  // Get the size of the file.
  size_t dataLength = file.tellg();
  // Move file pointer to the beginning of the file.
  file.seekg(0, std::ios::beg);
  // Create a vector to store the data.
  std::vector<uint8_t> fileData(dataLength);
  // Read the data.
  if (!file.read(reinterpret_cast<char *>(fileData.data()), dataLength)) {
    throw std::runtime_error("Error: Unable to read data from " + filePath);
  }
  file.close();

  // ---------------------------------------------------------------------------
  // 2. Determine the image format and decode the image data into MemRef.
  // ---------------------------------------------------------------------------
  // Determine the image format from the raw file data.
  determineFormat(fileData);
  if (this->imageFormat == ImageFormat::BMP) {
    bool success = decodeBMP(fileData);
    if (!success) {
      this->imageFormat = ImageFormat::ERROR;
      throw std::runtime_error("Failed to decode BMP file from " + filePath);
    };
  } else if (this->imageFormat == ImageFormat::PNG) {
    bool success = decodePNG(fileData);
    if (!success) {
      this->imageFormat = ImageFormat::ERROR;
      throw std::runtime_error("Failed to decode PNG file from " + filePath);
    };
  } else {
    throw std::runtime_error("Unsupported image file format.");
  }
}

// Determines the image format by inspecting the header of the file data.
template <typename T, std::size_t N>
void Image<T, N>::determineFormat(const std::vector<uint8_t> &fileData) {
  std::array<unsigned char, 8> pngHeader = {0x89, 0x50, 0x4E, 0x47,
                                            0x0D, 0x0A, 0x1A, 0x0A};
  if (fileData.size() > 2 && fileData[0] == 'B' && fileData[1] == 'M') {
    this->imageFormat = ImageFormat::BMP;
  } else if (fileData.size() > 7 &&
             std::memcmp(fileData.data(), pngHeader.data(), 8) == 0) {
    this->imageFormat = ImageFormat::PNG;
  } else {
    this->imageFormat = ImageFormat::ERROR;
  }
}

// BMP Image File Decoder
template <typename T, std::size_t N>
bool Image<T, N>::decodeBMP(const std::vector<uint8_t> &fileData) {
  // Check if the provided data is large enough to contain a minimal BMP header
  // (54 bytes).
  if (fileData.size() < 54) {
    throw std::runtime_error("Invalid BMP File: too small to contain header");
  }

  // Extract image information from BMP header
  this->width = *reinterpret_cast<const int32_t *>(&fileData[18]);
  this->height = *reinterpret_cast<const int32_t *>(&fileData[22]);
  this->bitDepth = *reinterpret_cast<const uint16_t *>(&fileData[28]);
  uint32_t compression = *reinterpret_cast<const uint32_t *>(&fileData[30]);
  size_t pixelDataOffset = *reinterpret_cast<const uint32_t *>(&fileData[10]);

  // Currently, only the BI_RGB (value 0) compression method is supported.
  if (compression != 0) {
    std::cerr << "Unsupported BMP file compression method." << std::endl;
    return false;
  }

  // Currently, only the NCHW format with 4 dimensions is supported.
  if (N == 4) {
    if (this->imageMode == ImageModes::DIP_GRAYSCALE) {
      // TODO: Add batch setting.
      this->sizes[0] = 1;
      this->sizes[1] = 1;
      this->sizes[2] = this->height;
      this->sizes[3] = this->width;
      this->setStrides();
      size_t size = this->product(this->sizes);
      this->allocated = (T *)malloc(sizeof(T) * size);
      this->aligned = this->allocated;
      // Fullfill data to memref container.
      size_t memrefIndex = 0;
      if (this->bitDepth == 32) {
        // BMP file is upside-down storage.
        for (size_t i = this->height; i > 0; i--) {
          for (size_t j = 0; j < this->width; j++) {
            // Locate the current pixel.
            size_t pixelIndex =
                pixelDataOffset + (((i - 1) * this->width) + j) * 4;
            // Extract the blue, green, and red value from the current pixel.
            int bluePixel =
                *reinterpret_cast<const uint8_t *>(&fileData[pixelIndex]);
            int greenPixel =
                *reinterpret_cast<const uint8_t *>(&fileData[pixelIndex + 1]);
            int redPixel =
                *reinterpret_cast<const uint8_t *>(&fileData[pixelIndex + 2]);
            // Calculate the gray scale value.
            int grayScaleValue = static_cast<int>(
                0.299 * redPixel + 0.587 * greenPixel + 0.114 * bluePixel);
            // Store the gray scale value into memref container.
            this->aligned[memrefIndex] =
                this->isNorm ? static_cast<T>(grayScaleValue) / 255
                             : static_cast<T>(grayScaleValue);
            memrefIndex++;
          }
        }
      } else {
        std::cerr << "Unsupported: " << this->bitDepth << "bit depth."
                  << std::endl;
        return false;
      }
    } else if (this->imageMode == ImageModes::DIP_RGB) {
      // TODO: Add batch setting.
      this->sizes[0] = 1;
      this->sizes[1] = 3;
      this->sizes[2] = this->height;
      this->sizes[3] = this->width;
      this->setStrides();
      size_t size = this->product(this->sizes);
      this->allocated = (T *)malloc(sizeof(T) * size);
      this->aligned = this->allocated;
      // Fullfill data to memref container.
      size_t memrefIndex = 0;
      size_t colorStride = this->height * this->width;
      if (this->bitDepth == 32) {
        // BMP file is upside-down storage.
        for (size_t i = height; i > 0; i--) {
          for (size_t j = 0; j < width; j++) {
            // Locate the current pixel.
            size_t pixelIndex = pixelDataOffset + (((i - 1) * width) + j) * 4;
            // Extract the blue, green, and red value from the current pixel.
            int bluePixel =
                *reinterpret_cast<const uint8_t *>(&fileData[pixelIndex]);
            int greenPixel =
                *reinterpret_cast<const uint8_t *>(&fileData[pixelIndex + 1]);
            int redPixel =
                *reinterpret_cast<const uint8_t *>(&fileData[pixelIndex + 2]);
            // Store the values into memref container as RGB order. (BGR -> RGB)
            this->aligned[memrefIndex] = this->isNorm
                                             ? static_cast<T>(redPixel) / 255
                                             : static_cast<T>(redPixel);
            this->aligned[memrefIndex + colorStride] =
                this->isNorm ? static_cast<T>(greenPixel) / 255
                             : static_cast<T>(greenPixel);
            this->aligned[memrefIndex + 2 * colorStride] =
                this->isNorm ? static_cast<T>(bluePixel) / 255
                             : static_cast<T>(bluePixel);
            memrefIndex++;
          }
        }
      } else {
        std::cerr << "Unsupported: " << this->bitDepth << "bit depth."
                  << std::endl;
        return false;
      }
    }
  } else {
    std::cerr << "Unsupported: " << N << " dimension layout." << std::endl;
    return false;
  }
  return true;
}

// PNG Image File Decoder
template <typename T, std::size_t N>
bool Image<T, N>::decodePNG(const std::vector<uint8_t> &fileData) {
  // Check if the provided data is large enough to contain a minimal PNG header
  // (33 bytes).
  if (fileData.size() < 33) {
    throw std::runtime_error("Invalid PNG File: too small to contain header");
  }

  // Extract image information from PNG header. Convert Big-Endian to
  // Little-Endian.
  this->width = (fileData[16] << 24) | (fileData[17] << 16) |
                (fileData[18] << 8) | fileData[19];
  this->height = (fileData[20] << 24) | (fileData[21] << 16) |
                 (fileData[22] << 8) | fileData[23];
  this->bitDepth = *reinterpret_cast<const uint8_t *>(&fileData[24]);
  int colorType = *reinterpret_cast<const uint8_t *>(&fileData[25]);
  uint8_t interlace = *reinterpret_cast<const uint8_t *>(&fileData[28]);

  // Currently, only the NCHW format with 4 dimensions is supported.
  if (N == 4) {
    // use libpng to read png image. Initialize libpng parameters
    png_structp png_ptr =
        png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if (!png_ptr) {
      std::cerr << "png_ptr creation failed" << std::endl;
      return false;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
      std::cerr << "png_infop creation failed" << std::endl;
      return false;
    }

    // Set jump point for error handling
    if (setjmp(png_jmpbuf(png_ptr))) {
      std::cerr << "error during PNG reading" << std::endl;
      // close PNG reading and free memory
      png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
      return false;
    }

    // copy filedata. Read image data from memory.
    std::vector<uint8_t> dataCopy = fileData;
    png_set_read_fn(
        png_ptr, &dataCopy,
        [](png_structp png_ptr, png_bytep data, png_size_t length) {
          std::vector<uint8_t> *fileData =
              static_cast<std::vector<uint8_t> *>(png_get_io_ptr(png_ptr));
          if (fileData->size() < length) {
            png_error(png_ptr, "Read error from memory");
          }
          std::copy(fileData->begin(), fileData->begin() + length, data);
          fileData->erase(fileData->begin(), fileData->begin() + length);
        });

    png_read_info(png_ptr, info_ptr);

    // Convert big or little Endian and convert 16 bits to 8 bits
    if (this->bitDepth == 16)
      png_set_strip_16(png_ptr);
    else if (!isBigEndian())
      png_set_swap(png_ptr);

    // Remove alpha channel
    if (colorType & PNG_COLOR_MASK_ALPHA)
      png_set_strip_alpha(png_ptr);

    // Convert palette to rgb
    if (colorType == PNG_COLOR_TYPE_PALETTE)
      png_set_palette_to_rgb(png_ptr);

    // Convert low depth grayscale to 8-bit grayscale
    if ((colorType & PNG_COLOR_MASK_COLOR) == 0 && this->bitDepth < 8)
#if (PNG_LIBPNG_VER_MAJOR * 10000 + PNG_LIBPNG_VER_MINOR * 100 +               \
         PNG_LIBPNG_VER_RELEASE >=                                             \
     10209) ||                                                                 \
    (PNG_LIBPNG_VER_MAJOR == 1 && PNG_LIBPNG_VER_MINOR == 0 &&                 \
     PNG_LIBPNG_VER_RELEASE >= 18)
      png_set_expand_gray_1_2_4_to_8(png_ptr);
#else
      png_set_gray_1_2_4_to_8(png_ptr);
#endif

    // Processing interleaved PNG images
    if (interlace)
      png_set_interlace_handling(png_ptr);

    if (this->imageMode == ImageModes::DIP_GRAYSCALE) {
      // TODO: Add batch setting.
      this->sizes[0] = 1;
      this->sizes[1] = 1;
      this->sizes[2] = this->height;
      this->sizes[3] = this->width;
      this->setStrides();
      size_t size = this->product(this->sizes);
      this->allocated = (T *)malloc(sizeof(T) * size);
      this->aligned = this->allocated;

      // RGB->Gray
      if ((colorType & PNG_COLOR_MASK_COLOR) ||
          (colorType == PNG_COLOR_TYPE_PALETTE))
        png_set_rgb_to_gray(png_ptr, 1, 0.299, 0.587);

      // Update reading setting
      png_read_update_info(png_ptr, info_ptr);

      // Allocate memory for libpng to read images
      std::vector<uint8_t> imgData(this->height * this->width);
      std::vector<uint8_t *> row_pointers(this->height);
      for (size_t y = 0; y < this->height; ++y) {
        row_pointers[y] = imgData.data() + y * this->width;
      }

      // Reading image
      png_read_image(png_ptr, row_pointers.data());

      // Fullfill data to memref container.
      for (size_t i = 0; i < this->height; i++)
        for (size_t j = 0; j < this->width; j++) {
          size_t memrefIndex = i * this->width + j;
          this->aligned[memrefIndex] =
              this->isNorm ? static_cast<T>(imgData[memrefIndex]) / 255
                           : static_cast<T>(imgData[memrefIndex]);
          ;
        }
    } else if (this->imageMode == ImageModes::DIP_RGB) {
      // TODO: Add batch setting.
      this->sizes[0] = 1;
      this->sizes[1] = 3;
      this->sizes[2] = this->height;
      this->sizes[3] = this->width;
      this->setStrides();
      size_t size = this->product(this->sizes);
      std::cout << "size" << size << std::endl;
      this->allocated = (T *)malloc(sizeof(T) * size);
      this->aligned = this->allocated;
      size_t colorStride = this->height * this->width;

      // Gray->RGB
      if (colorType & PNG_COLOR_TYPE_GRAY)
        png_set_gray_to_rgb(png_ptr);

      // Update reading setting
      png_read_update_info(png_ptr, info_ptr);

      // Allocate memory for libpng to read images
      std::vector<uint8_t> imgData(this->height * this->width * 3);
      std::vector<uint8_t *> row_pointers(this->height);
      for (size_t y = 0; y < this->height; ++y) {
        row_pointers[y] = imgData.data() + y * this->width * 3;
      }

      // Reading image
      png_read_image(png_ptr, row_pointers.data());

      // Separate pixel data by channel
      size_t memrefIndex = 0;
      for (size_t i = 0; i < this->height; i++)
        for (size_t j = 0; j < this->width; j++) {
          // Locate the current pixel.
          size_t pixelIndex = ((i * width) + j) * 3;
          // Extract the red, green, and blue value from the current pixel.
          int redPixel =
              *reinterpret_cast<const uint8_t *>(&imgData[pixelIndex]);
          int greenPixel =
              *reinterpret_cast<const uint8_t *>(&imgData[pixelIndex + 1]);
          int bluePixel =
              *reinterpret_cast<const uint8_t *>(&imgData[pixelIndex + 2]);
          // Store the values into memref container as RGB order.
          this->aligned[memrefIndex] = this->isNorm
                                           ? static_cast<T>(redPixel) / 255
                                           : static_cast<T>(redPixel);
          this->aligned[memrefIndex + colorStride] =
              this->isNorm ? static_cast<T>(greenPixel) / 255
                           : static_cast<T>(greenPixel);
          this->aligned[memrefIndex + 2 * colorStride] =
              this->isNorm ? static_cast<T>(bluePixel) / 255
                           : static_cast<T>(bluePixel);
          memrefIndex++;
        }
    }

    // close PNG reading and free memory
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  } else {
    std::cerr << "Unsupported: " << N << " dimension layout." << std::endl;
    return false;
  }
  return true;
}

} // namespace dip

#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMGCONTAINER
