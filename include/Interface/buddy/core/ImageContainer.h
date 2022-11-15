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

#ifndef INTERFACE_BUDDY_CORE_IMAGECONTAINER
#define INTERFACE_BUDDY_CORE_IMAGECONTAINER

#include "Interface/buddy/core/Container.h"
#include <opencv2/opencv.hpp>

// Image container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
template <typename T, size_t N> class Img : public MemRef<T, N> {
public:
  Img(cv::Mat image);
};
template <typename T, size_t N>class Stb_img <T,N>:public MemRef<T,N>
{
  private:
    int Rows;
    int Cols;
    int Channels;
    unsigned char *Stbi_data;
  public:
      //1、有参构造函数,输入文件名，创建一个图像容器
    Stb_img(const string &filename)
    {
      int x, y, n;
      unsigned char *data = stbi_load(filename.c_str(), &x, &y, &n, 0);//读取图像数据首地址，像素数据RGB形式存储在缓存区
      if (!data) 
      {
        std::cout << "图片读取错误" << endl;
        return;
      } 
      else if (n == 1) 
      {
        assert((N == 2) &&"Input image type does not match the selected dimension.");
        this->Rows = y;
        this->Cols = x;
        this->Channels = n;
        this->Stbi_data = data;
        this->allocated = new (*Stbi_data); // unsigned char类型转换成(float)类型是否存在问题（讨论）
        this->sizes[0] = Rows;
        this->sizes[1] = Cols;
        this->size = Rows * Cols * N;
      }
    }
 //析构函数,给维护的图片指针置空
  public:
    ~Stb_img()
    {
      if (this->Stbi_data) 
      {
        delete Stbi_data;
        Stbi_data = NULL;
      }
    }
    // getter and setter
    unsigned int GetWidth() const { return Cols; }
    unsigned int GetHeight() const { return Rows; }
    unsigned int Ge tChannels() const { return Channels; }
    unsigned int GetSize() const { return Rows * Cols * Channels; }
    unsigned char *Getidata() const { return Stbi_data }
    //2、实现写入一个图片；
    //3、实现图片旋转；
};

#include "Interface/core/ImageContainer.cpp"

#endif // INTERFACE_BUDDY_CORE_IMAGECONTAINER
