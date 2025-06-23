#include <buddy/Core/Container.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>


extern "C" void _mlir_ciface_forward(MemRef<__fp16, 1> *output,
                                     MemRef<__fp16, 1> *arg0,
                                     MemRef<__fp16, 1> *arg1);

int main() {
  const std::string title = "FP16 AddOp Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  intptr_t size[1] = {10};  //  size of the input and output tensors

  MemRef<__fp16, 1> output(size);
  MemRef<__fp16, 1> arg0(size);
  MemRef<__fp16, 1> arg1(size);

  //  initialize test data
  __fp16 arg0_data[10] = {0.0f, -0.0f, 65504.0f, -65504.0f, 3.75f, 5.5f, 1.5f, 0.125f, 0.0625f, 0.1f};
  __fp16 arg1_data[10] = {-0.0f, -0.0f, -65504.0f, 65504.0f, 3.75f, 5.5f, 0.125f, 1.5f, 0.0625f, 0.1f};
  std::memcpy(arg0.getData(), arg0_data, sizeof(arg0_data));
  std::memcpy(arg1.getData(), arg1_data, sizeof(arg1_data));

  _mlir_ciface_forward(&output, &arg0, &arg1);
  auto out = output.getData();

  //  test the result
  for (int i = 0; i < 10; i++) {
    uint16_t binaryRepresentation;
    std::memcpy(&binaryRepresentation, &out[i], sizeof(__fp16));
    std::bitset<16> bits(binaryRepresentation);
    std::cout << bits << " " << arg0_data[i] << " + " << arg1_data[i] << " = "<< out[i] << std::endl;
  }

  return 0;
}