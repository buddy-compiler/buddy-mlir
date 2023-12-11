#include <pybind11/pybind11.h>
#include <buddy/Core/Container.h>

extern "C" void _mlir_ciface_forward(MemRef<float, 2> *, MemRef<float, 2> *,
                                     MemRef<float, 2> *);

int add(int i, int j) {
    return i + j;
}



PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
}