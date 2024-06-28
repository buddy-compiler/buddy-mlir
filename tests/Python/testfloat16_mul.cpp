//buddy-mlir/tests/Python/testfloat16_mul.cpp
#include <buddy/Core/Container.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include<iomanip>
extern "C" void _mlir_ciface_forward(MemRef<uint16_t, 1> *, MemRef<uint16_t, 1>*);

// f16 -> f32
float f162f32(uint16_t &f16){
	uint32_t sign = (f16 >> 15) & 0x1;
	uint32_t exponent = ((f16 >> 10) & 0x1f) - 15 + 127;
	uint32_t mantissa = (f16 & 0x3ff) << 13;
	uint32_t bits = (sign << 31) | (exponent << 23) | mantissa;
	float f = *((float*) &bits);
	return f;
}

// f32 -> f16
uint16_t f322f16(float& input_fp32){
	float f = input_fp32;
	uint32_t bits = *((uint32_t*) &f);
	uint16_t sign = (bits >> 31) & 0x1;
	uint16_t exponent = ((bits >> 23) & 0xff) - 127 + 15;
	uint16_t mantissa = (bits & 0x7fffff) >> 13;
	uint16_t f16 = (sign << 15) | (exponent << 10) | mantissa;
	return f16;
}
int main(){
	MemRef<uint16_t, 1> a = MemRef<uint16_t, 1>({1}); //operand
	MemRef<uint16_t, 1> res = MemRef<uint16_t, 1>({1}); // result
	float f1 ;
	std::cout << "pleace enter a num: ";
	std::cin >> f1;
	*a.getData() = f322f16(f1);
	_mlir_ciface_forward( &res, &a);
	std::cout<<f1<<" mul 2 = " << f162f32(*res.getData()) <<std::endl;
	return 0;
}
