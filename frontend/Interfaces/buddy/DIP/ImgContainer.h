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

namespace dip {
enum ImageModes {
  DIP_GRAYSCALE = 0,
  DIP_RGB = 1,
};

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
  } else {
    throw std::runtime_error("Unsupported image file format.");
  }
}

// Determines the image format by inspecting the header of the file data.
template <typename T, std::size_t N>
void Image<T, N>::determineFormat(const std::vector<uint8_t> &fileData) {
  if (fileData.size() > 2 && fileData[0] == 'B' && fileData[1] == 'M') {
    this->imageFormat = ImageFormat::BMP;
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

  // Currently, only the BI_RGB (value 0) or BI_BITFIELDS (value 3) compression
  // method is supported.
  if (compression != 0 && compression != 3) {
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
      } else if (this->bitDepth == 24) {
        // BMP file is upside-down storage.
        for (size_t i = this->height; i > 0; i--) {
          for (size_t j = 0; j < this->width; j++) {
            // Locate the current pixel.
            size_t pixelIndex =
                pixelDataOffset + (((i - 1) * this->width) + j) * 3;
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
      } else if (this->bitDepth == 16) {
        // BMP file is upside-down storage.
        for (size_t i = this->height; i > 0; i--) {
          for (size_t j = 0; j < this->width; j++) {
            // Locate the current pixel.
            size_t pixelIndex =
                pixelDataOffset + (((i - 1) * this->width) + j) * 2;
            // Extract the 16-bit pixel value
            uint16_t pixelValue =
                *reinterpret_cast<const uint16_t *>(&fileData[pixelIndex]);

            int redPixel, greenPixel, bluePixel;
            if (compression == 3) {
              // Extract individual color components (assuming RGB565 format)
              redPixel = (pixelValue >> 11) & 0x1F;
              greenPixel = (pixelValue >> 5) & 0x3F;
              bluePixel = pixelValue & 0x1F;

              // Expand to 8-bit per channel
              redPixel = (redPixel << 3) | (redPixel >> 2);
              greenPixel = (greenPixel << 2) | (greenPixel >> 4);
              bluePixel = (bluePixel << 3) | (bluePixel >> 2);
            } else {
              // Extract individual color components for 5-5-5 format
              redPixel = (pixelValue >> 10) & 0x1F;
              greenPixel = (pixelValue >> 5) & 0x1F;
              bluePixel = pixelValue & 0x1F;

              // Expand to 8-bit per channel
              redPixel = (redPixel << 3) | (redPixel >> 2);
              greenPixel = (greenPixel << 3) | (greenPixel >> 2);
              bluePixel = (bluePixel << 3) | (bluePixel >> 2);
            }
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
      } else if (this->bitDepth == 24) {
        // BMP file is upside-down storage.
        for (size_t i = height; i > 0; i--) {
          for (size_t j = 0; j < width; j++) {
            // Locate the current pixel.
            size_t pixelIndex = pixelDataOffset + (((i - 1) * width) + j) * 3;
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
      } else if (this->bitDepth == 16) {
        // BMP file is upside-down storage.
        for (size_t i = height; i > 0; i--) {
          for (size_t j = 0; j < width; j++) {
            // Locate the current pixel.
            size_t pixelIndex = pixelDataOffset + (((i - 1) * width) + j) * 2;
            // Extract the 16-bit pixel value
            uint16_t pixelValue =
                *reinterpret_cast<const uint16_t *>(&fileData[pixelIndex]);

            int redPixel, greenPixel, bluePixel;
            if (compression == 3) {
              // Extract individual color components (assuming RGB565 format)
              redPixel = (pixelValue >> 11) & 0x1F;
              greenPixel = (pixelValue >> 5) & 0x3F;
              bluePixel = pixelValue & 0x1F;

              // Expand to 8-bit per channel
              redPixel = (redPixel << 3) | (redPixel >> 2);
              greenPixel = (greenPixel << 2) | (greenPixel >> 4);
              bluePixel = (bluePixel << 3) | (bluePixel >> 2);
            } else {
              // Extract individual color components for 5-5-5 format
              redPixel = (pixelValue >> 10) & 0x1F;
              greenPixel = (pixelValue >> 5) & 0x1F;
              bluePixel = pixelValue & 0x1F;

              // Expand to 8-bit per channel
              redPixel = (redPixel << 3) | (redPixel >> 2);
              greenPixel = (greenPixel << 3) | (greenPixel >> 2);
              bluePixel = (bluePixel << 3) | (bluePixel >> 2);
            }

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

} // namespace dip

#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMGCONTAINER
