//===- ImageContainer.h ---------------------------------------------------===//
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
// Image container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER
#define FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER

#include "buddy/Core/Container.h"
#include <cassert>
#include <opencv2/opencv.hpp>

// Image container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
// - image represents the OpenCV Mat object.
// - norm indicates whether to perform normalization, and the normalization is
//   disabled by default.
template <typename T, size_t N> class Img : public MemRef<T, N> {
public:
  //Img(cv::Mat image, intptr_t sizes[N] = nullptr, bool norm = false);
  /*
		Img类默认构造函数
	*/    
    Img();
    /*
    	@rows:二维数组的行数
    	@cols:二维数组的列数
    	@type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或
CV_8UC (n),……， CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
    */
    Img(int rows , int cols , int type);
    /*
    	@size：二维数组的大小 类Size(cols,rows)
    	@type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或
CV_8UC (n),……， CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
    */
    Img(Size size, int type);
    /*
    	@rows:二维数组的行数
    	@cols:二维数组的列数
    	@type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或
CV_8UC (n),……， CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
		@s:使用指定的Scalar颜色填充二维数组
    */
    Img(int rows , int cols ,int type , const Scalar& s);
    /*
    	@size：二维数组的大小 类Size(cols,rows)
    	@type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或
CV_8UC (n),……， CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
		@s:使用指定的Scalar颜色填充二维数组
    */
    Img(Size size , int type , const Scalar & s);
    /*
    	@ndims:数组的维度
    	@sizes：用来指定n维数组的形状
    	@type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或
CV_8UC (n),……， CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
    */
    Img(int ndims , const int*sizes , int type);
    /*
    	@sizes:使用vector容器存储n维数组的形状
    	@type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或
CV_8UC (n),……， CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
    */
    Img(const std::vector<int>& sizes, int type);
	/*
		@ndims:数组的维度
		@sizes：用来指定n维数组的形状
		@type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或
CV_8UC (n),……， CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
		@s:使用指定的Scalar颜色填充二维数组
	*/ 
    Img(int ndims, const int* sizes, int type, const Scalar& s);
    /*
    	@sizes:使用vector容器存储n维数组的形状
    	@type:数组类型 使用CV_8UC1，…， CV_64FC4创建1-4通道矩阵，或
CV_8UC (n),……， CV_64FC(n)创建多通道(最多CV_CN_MAX通道)矩阵。
		@s:使用指定的Scalar颜色填充二维数组
    */
    Img(const std::vector<int>& sizes, int type, const Scalar& s);
	/*
		@m:赋值构造函数，并不发生拷贝。当用户使用这个构造函数进行构造时，改变该矩阵同时也会改变用来构建该矩阵的源矩阵。如果用户想拥有独立的内存空间用来拷贝新矩阵，可以使用 Img::clone() 
	*/
    Img(const Img& m);
    /*
    	@data:指向用户数据的指针(存储图像数据的地址)，Img不会通过data和step来为数据分配内存，仅仅利用header来指向数据。
    	@step:每个矩阵行所占有的字节数。这个值应该包含行尾的填充字节.如果缺省该参数，则默认没有填充字节且实际的步长由cols*elemSize()决定. 
    */
    Img(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);

    Img(Size size, int type, void* data, size_t step=AUTO_STEP);

    Img(int ndims, const int* sizes, int type, void* data, const size_t* steps=0);

    Img(const std::vector<int>& sizes, int type, void* data, const size_t* steps=0);
	/*
		@rowRange:要选取m行的范围，范围为左闭右开。可以使用Range::all表示所有行。
		@colRange:要选取m列的范围，可以使用Range::all表示所有列。	
	*/
    Img(const Img& m, const Range& rowRange, const Range& colRange=Range::all());
	/*
		@roi:感兴趣的区域
	*/
    Img(const Img& m, const Rect& roi);
	
    ~Img();
  
/*
    	成员函数。
    */
    void create(int rows , int cols ,int type);
    void create(Size size , int type);
    void create(int ndims , const int * sizes , int type);
    void create(const std::vector<int>&sizes , int type);
    void addref();
    void release();
    void deallocate();
    
    static bool load(const String& filename , int flags , Img& img);
    static bool save(const String& filename);
    
    template<typename _Tp> void push_back(const _Tp& elem);
    template<typename _Tp> void push_back(const Mat_<_Tp>& elem);
    template<typename _Tp> void push_back(const std::vector<_Tp>& elem);
    int type() const;
  
    /*
    	成员数据。
    */
    int flags;
    //! 矩阵维度, >= 2
    int dims;
    //! 当矩阵有多个维度时，行数及列数或者(-1, -1) 
    int rows, cols;
    //! 指向数据的指针
    T* data;
};

/*
// Image Constructor from OpenCV Mat.
template <typename T, size_t N>
Img<T, N>::Img(cv::Mat image, intptr_t sizes[N], bool norm) : MemRef<T, N>() {
  if (image.channels() == 1) {
    assert((N == 2) && "For gray images, the number of dimensions must be 2.");
  } else if (image.channels() == 3) {
    assert((N == 4) && "For RGB images, the number of dimensions must be 4, "
                       "either in NHWC or NCHW layout.");
  } else {
    std::cerr << "Only 2-channel gray images and 3-channel RGB images are "
                 "supported, but got images' channel equal to "
              << image.channels() << "." << std::endl;
  }
  // Use default layout setting.
  if (sizes == nullptr) {
    // The size of the gray image is represented by height and width by default.
    if (N == 2) {
      this->sizes[0] = image.rows;
      this->sizes[1] = image.cols;
    }
    // For RGB images, use NHWC layout by default.
    else if (N == 4) {
      this->sizes[0] = 1;
      this->sizes[1] = image.rows;
      this->sizes[2] = image.cols;
      this->sizes[3] = 3;
    }
  } else {
    // Use custom layout setting.
    for (size_t i = 0; i < N; i++) {
      this->sizes[i] = sizes[i];
    }
  }
  this->size = this->product(this->sizes);
  this->setStrides();
  this->allocated = new T[this->size];
  this->aligned = this->allocated;
  this->loadImg(image, norm);
}

template <typename T, size_t N>
void Img<T, N>::loadImg(cv::Mat image, bool norm) {
  // Load gray image data from OpenCV Mat.
  if (N == 2) {
    size_t k = 0;
    for (int i = 0; i < this->sizes[0]; i++) {
      for (int j = 0; j < this->sizes[1]; j++) {
        if (norm) {
          this->aligned[k] = (T)image.at<uchar>(i, j) / 255;
        } else {
          this->aligned[k] = (T)image.at<uchar>(i, j);
        }
        k++;
      }
    }
  } else if (N == 4) {
    // Detect NHWC layout of RGB image data.
    if (this->sizes[1] == image.rows && this->sizes[2] == image.cols &&
        this->sizes[3] == 3) {
      size_t k = 0;
      for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
          for (int color = 0; color < 3; color++) {
            if (norm) {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color] / 255;
            } else {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color];
            }
            k++;
          }
        }
      }
    }
    // Detect NCHW layout of RGB image data.
    else if (this->sizes[2] == image.rows && this->sizes[3] == image.cols &&
             this->sizes[1] == 3) {
      size_t k = 0;
      for (int color = 0; color < 3; color++) {
        for (int i = 0; i < image.rows; i++) {
          for (int j = 0; j < image.cols; j++) {
            if (norm) {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color] / 255;
            } else {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color];
            }
            k++;
          }
        }
      }
    } else {
      std::cerr << "RGB images must be arranged in either NHWC or NCHW layout."
                << std::endl;
    }
  }
}
*/
#endif // FRONTEND_INTERFACES_BUDDY_DIP_IMAGECONTAINER
