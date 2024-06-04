#include <buddy/Core/Container.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include<iomanip>

extern "C" void _mlir_ciface_forward(MemRef<uint16_t, 2> *, MemRef<uint16_t, 2> *, MemRef<uint16_t, 2> *);

// f16 -> f32
float f162f32(uint16_t f16){
 uint32_t sign = (f16 >> 15) & 0x1;
 uint32_t exponent = ((f16 >> 10) & 0x1f) - 15 + 127;
 uint32_t mantissa = (f16 & 0x3ff) << 13;
 uint32_t bits = (sign << 31) | (exponent << 23) | mantissa;
 float f = *((float*) &bits);
 return f;
}

// f32 -> f16
uint16_t f322f16(float input_fp32){
 float f = input_fp32;
 uint32_t bits = *((uint32_t*) &f);
 uint16_t sign = (bits >> 31) & 0x1;
 uint16_t exponent = ((bits >> 23) & 0xff) - 127 + 15;
 uint16_t mantissa = (bits & 0x7fffff) >> 13;
 uint16_t f16 = (sign << 15) | (exponent << 10) | mantissa;
 return f16;
}

int main(){
        MemRef<uint16_t, 2> a = MemRef<uint16_t, 2>({10, 5});  //operand
        MemRef<uint16_t, 2> b = MemRef<uint16_t, 2>({3, 3});
        MemRef<uint16_t, 2> res = MemRef<uint16_t, 2>({9, 6});  //result
        float fnum = 1.1;
        for (int i = 0; i < 10 * 5; i++){
                *(a.getData() + i) = f322f16(fnum);
                fnum += 1;
        }

        int16_t inum = 1;
        for (int i = 0; i < 3 * 3; i++){
                if ((int)inum % 2 == 0)
                        *(b.getData() + i*2) = (uint16_t)(inum - 2);
                else
                        *(b.getData() + i*2) = (uint16_t)inum;
                inum += 1;
        }

        _mlir_ciface_forward( &res, &a, &b);

        int index = 0;
        std::cout<<"[";   //print result
        for(int i = 0; i < 3; i++) {
                std::cout<<"[";
                for (int j = 0; j < 3; j++) {
                        std::cout<<"[";
                        for (int t = 0; t < 5; t++){
                                std::cout<<f162f32(*(res.getData() + index))<<", ";
                                index++;
                        }
                        std::cout<<"],";
                        if (j != 2)
                                std::cout<<std::endl;
                }
                std::cout<<"],";
                if (i != 2) {
                        std::cout<<std::endl;
                        std::cout<<std::endl;
                }

        }
        std::cout<<"]";
        std::cout<<std::endl;

        return 0;
}

