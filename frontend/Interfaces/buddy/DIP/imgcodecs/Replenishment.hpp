
#ifndef REPLEISHMENT
#define REPLEISHMENT

#include <cassert>
#include <memory> // std::shared_ptr
#include <string>
#include <type_traits> // std::enable_if

using namespace std;
typedef unsigned long ulong;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef string String;

#define CV_CN_MAX 512
#define CV_CN_SHIFT 3
#define CV_DEPTH_MAX (1 << CV_CN_SHIFT)
#define CV_MAT_CN_MASK ((CV_CN_MAX - 1) << CV_CN_SHIFT)
#define CV_MAT_CN(flags) ((((flags)&CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)

#define CV_8U 0
#define CV_8S 1
#define CV_16U 2
#define CV_16S 3
#define CV_32S 4
#define CV_32F 5
#define CV_64F 6
#define CV_16F 7

#define CV_MAT_DEPTH_MASK (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags) ((flags)&CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth, cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U, 1)
#define CV_8UC2 CV_MAKETYPE(CV_8U, 2)
#define CV_8UC3 CV_MAKETYPE(CV_8U, 3)
#define CV_8UC4 CV_MAKETYPE(CV_8U, 4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U, (n))

#define CV_8SC1 CV_MAKETYPE(CV_8S, 1)
#define CV_8SC2 CV_MAKETYPE(CV_8S, 2)
#define CV_8SC3 CV_MAKETYPE(CV_8S, 3)
#define CV_8SC4 CV_MAKETYPE(CV_8S, 4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S, (n))

#define CV_16UC1 CV_MAKETYPE(CV_16U, 1)
#define CV_16UC2 CV_MAKETYPE(CV_16U, 2)
#define CV_16UC3 CV_MAKETYPE(CV_16U, 3)
#define CV_16UC4 CV_MAKETYPE(CV_16U, 4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U, (n))

#define CV_16SC1 CV_MAKETYPE(CV_16S, 1)
#define CV_16SC2 CV_MAKETYPE(CV_16S, 2)
#define CV_16SC3 CV_MAKETYPE(CV_16S, 3)
#define CV_16SC4 CV_MAKETYPE(CV_16S, 4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S, (n))

#define CV_32SC1 CV_MAKETYPE(CV_32S, 1)
#define CV_32SC2 CV_MAKETYPE(CV_32S, 2)
#define CV_32SC3 CV_MAKETYPE(CV_32S, 3)
#define CV_32SC4 CV_MAKETYPE(CV_32S, 4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S, (n))

#define CV_32FC1 CV_MAKETYPE(CV_32F, 1)
#define CV_32FC2 CV_MAKETYPE(CV_32F, 2)
#define CV_32FC3 CV_MAKETYPE(CV_32F, 3)
#define CV_32FC4 CV_MAKETYPE(CV_32F, 4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F, (n))

#define CV_64FC1 CV_MAKETYPE(CV_64F, 1)
#define CV_64FC2 CV_MAKETYPE(CV_64F, 2)
#define CV_64FC3 CV_MAKETYPE(CV_64F, 3)
#define CV_64FC4 CV_MAKETYPE(CV_64F, 4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F, (n))

#define CV_16FC1 CV_MAKETYPE(CV_16F, 1)
#define CV_16FC2 CV_MAKETYPE(CV_16F, 2)
#define CV_16FC3 CV_MAKETYPE(CV_16F, 3)
#define CV_16FC4 CV_MAKETYPE(CV_16F, 4)
#define CV_16FC(n) CV_MAKETYPE(CV_16F, (n))

#define CV_ELEM_SIZE1(type) ((0x28442211 >> CV_MAT_DEPTH(type) * 4) & 15)
#define CV_ELEM_SIZE(type) (CV_MAT_CN(type) * CV_ELEM_SIZE1(type))

//! Imread flags
enum ImreadModes {
  IMREAD_UNCHANGED =
      -1, //!< If set, return the loaded image as is (with alpha
          //!< channel,otherwise it gets cropped). Ignore EXIF orientation.
  IMREAD_GRAYSCALE = 0, //!< If set, always convert image to the single channel
                        //!< grayscale image (codec internal conversion).
  IMREAD_COLOR =
      1, //!< If set, always convert image to the 3 channel BGR color image.
  IMREAD_ANYDEPTH =
      2, //!< If set, return 16-bit/32-bit image when the input has the
         //!< corresponding depth, otherwise convert it to 8-bit.
  IMREAD_ANYCOLOR =
      4, //!< If set, the image is read in any possible color format.
  IMREAD_LOAD_GDAL = 8, //!< If set, use the gdal driver for loading the image.
  IMREAD_REDUCED_GRAYSCALE_2 =
      16, //!< If set, always convert image to the single channel grayscale
          //!< image and the image size reduced 1/2.
  IMREAD_REDUCED_COLOR_2 =
      17, //!< If set, always convert image to the 3 channel BGR color image and
          //!< the image size reduced 1/2.
  IMREAD_REDUCED_GRAYSCALE_4 =
      32, //!< If set, always convert image to the single channel grayscale
          //!< image and the image size reduced 1/4.
  IMREAD_REDUCED_COLOR_4 =
      33, //!< If set, always convert image to the 3 channel BGR color image and
          //!< the image size reduced 1/4.
  IMREAD_REDUCED_GRAYSCALE_8 =
      64, //!< If set, always convert image to the single channel grayscale
          //!< image and the image size reduced 1/8.
  IMREAD_REDUCED_COLOR_8 =
      65, //!< If set, always convert image to the 3 channel BGR color image and
          //!< the image size reduced 1/8.
  IMREAD_IGNORE_ORIENTATION = 128 //!< If set, do not rotate the image according
                                  //!< to EXIF's orientation flag.
};

//! Imwrite flags
enum ImwriteFlags {
  IMWRITE_JPEG_QUALITY =
      1, //!< For JPEG, it can be a quality from 0 to 100 (the higher is the
         //!< better). Default value is 95.
  IMWRITE_JPEG_PROGRESSIVE =
      2, //!< Enable JPEG features, 0 or 1, default is False.
  IMWRITE_JPEG_OPTIMIZE =
      3, //!< Enable JPEG features, 0 or 1, default is False.
  IMWRITE_JPEG_RST_INTERVAL =
      4, //!< JPEG restart interval, 0 - 65535, default is 0 - no restart.
  IMWRITE_JPEG_LUMA_QUALITY =
      5, //!< Separate luma quality level, 0 - 100, default is 0 - don't use.
  IMWRITE_JPEG_CHROMA_QUALITY =
      6, //!< Separate chroma quality level, 0 - 100, default is 0 - don't use.
  IMWRITE_PNG_COMPRESSION =
      16, //!< For PNG, it can be the compression level from 0 to 9. A higher
          //!< value means a smaller size and longer compression time. If
          //!< specified, strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT
          //!< (Z_DEFAULT_STRATEGY). Default value is 1 (best speed setting).
  IMWRITE_PNG_STRATEGY =
      17, //!< One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
  IMWRITE_PNG_BILEVEL = 18, //!< Binary level PNG, 0 or 1, default is 0.
  IMWRITE_PXM_BINARY = 32,  //!< For PPM, PGM, or PBM, it can be a binary format
                            //!< flag, 0 or 1. Default value is 1.
  IMWRITE_EXR_TYPE = (3 << 4) + 0,
  /* 48 */ //!< override EXR storage type (FLOAT (FP32) is default)
  IMWRITE_WEBP_QUALITY =
      64, //!< For WEBP, it can be a quality from 1 to 100 (the higher is the
          //!< better). By default (without any parameter) and for quality above
          //!< 100 the lossless compression is used.
  IMWRITE_PAM_TUPLETYPE =
      128, //!< For PAM, sets the TUPLETYPE field to the corresponding string
           //!< value that is defined for the format
  IMWRITE_TIFF_RESUNIT =
      256, //!< For TIFF, use to specify which DPI resolution unit to set; see
           //!< libtiff documentation for valid values
  IMWRITE_TIFF_XDPI = 257, //!< For TIFF, use to specify the X direction DPI
  IMWRITE_TIFF_YDPI = 258, //!< For TIFF, use to specify the Y direction DPI
  IMWRITE_TIFF_COMPRESSION =
      259, //!< For TIFF, use to specify the image compression scheme. See
           //!< libtiff for integer constants corresponding to compression
           //!< formats. Note, for images whose depth is CV_32F, only libtiff's
           //!< SGILOG compression scheme is used. For other supported depths,
           //!< the compression scheme can be specified by this flag; LZW
           //!< compression is the default.
  IMWRITE_JPEG2000_COMPRESSION_X1000 =
      272 //!< For JPEG2000, use to specify the target compression rate
          //!< (multiplied by 1000). The value can be from 0 to 1000. Default is
          //!< 1000.
};

enum ImwriteEXRTypeFlags {
  /*IMWRITE_EXR_TYPE_UNIT = 0, //!< not supported */
  IMWRITE_EXR_TYPE_HALF = 1, //!< store as HALF (FP16)
  IMWRITE_EXR_TYPE_FLOAT = 2 //!< store as FP32 (default)
};

//! Imwrite PNG specific flags used to tune the compression algorithm.
/** These flags will be modify the way of PNG image compression and will be
passed to the underlying zlib processing stage.

-   The effect of IMWRITE_PNG_STRATEGY_FILTERED is to force more Huffman coding
and less string matching; it is somewhat intermediate between
IMWRITE_PNG_STRATEGY_DEFAULT and IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY.
-   IMWRITE_PNG_STRATEGY_RLE is designed to be almost as fast as
IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, but give better compression for PNG image
data.
-   The strategy parameter only affects the compression ratio but not the
correctness of the compressed output even if it is not set appropriately.
-   IMWRITE_PNG_STRATEGY_FIXED prevents the use of dynamic Huffman codes,
allowing for a simpler decoder for special applications.
*/
enum ImwritePNGFlags {
  IMWRITE_PNG_STRATEGY_DEFAULT = 0, //!< Use this value for normal data.
  IMWRITE_PNG_STRATEGY_FILTERED =
      1, //!< Use this value for data produced by a filter (or
         //!< predictor).Filtered data consists mostly of small values with a
         //!< somewhat random distribution. In this case, the compression
         //!< algorithm is tuned to compress them better.
  IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY =
      2, //!< Use this value to force Huffman encoding only (no string match).
  IMWRITE_PNG_STRATEGY_RLE = 3, //!< Use this value to limit match distances to
                                //!< one (run-length encoding).
  IMWRITE_PNG_STRATEGY_FIXED =
      4 //!< Using this value prevents the use of dynamic Huffman codes,
        //!< allowing for a simpler decoder for special applications.
};

//! Imwrite PAM specific tupletype flags used to define the 'TUPETYPE' field of
//! a PAM file.
enum ImwritePAMFlags {
  IMWRITE_PAM_FORMAT_NULL = 0,
  IMWRITE_PAM_FORMAT_BLACKANDWHITE = 1,
  IMWRITE_PAM_FORMAT_GRAYSCALE = 2,
  IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA = 3,
  IMWRITE_PAM_FORMAT_RGB = 4,
  IMWRITE_PAM_FORMAT_RGB_ALPHA = 5,
};
class _Size {
public:
  _Size(){};
  _Size(int _width, int _height) : width(_width), height(_height) {}
  inline _Size &operator=(const _Size &rhs) {
    this->width = rhs.width;
    this->height = rhs.height;
    return *this;
  }
  _Size &operator+=(const _Size &rhs) {
    width += rhs.width;
    height += rhs.height;
    return *this;
  }
  bool operator==(const _Size &rhs) {
    return width == rhs.width && height == rhs.height;
  }
  bool operator!=(const _Size &rhs) { return !(*this == rhs); }
  int width = 0;
  int height = 0;
};

#endif // Repleishment_HPP
