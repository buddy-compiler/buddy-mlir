#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace buddy;

extern "C" void _mlir_ciface_forward(MemRef<_Float16, 3> *result, MemRef<float, 3> *params, MemRef<float, 3> *input);

std::vector<std::vector<float>> readDataFromFile(const std::string &filePath, int rows, int cols) {
    std::ifstream file(filePath);
    std::vector<std::vector<float>> data;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::vector<float> row;
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                row.push_back(value);
            }
            if (row.size() == cols) {
                data.push_back(row);
            } else {
                std::cerr << "Row size does not match expected cols: " << row.size() << " vs " << cols << std::endl;
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filePath << std::endl;
    }
    if (data.size() != rows) {
        std::cerr << "Data size does not match expected rows: " << data.size() << " vs " << rows << std::endl;
    }
    return data;
}

void printVector(const std::vector<float> &vec, const std::string &name) {
    std::cout << name << ": ";
    for (const auto &val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}


int main() {
    std::string inputFilePath = "/home/xujiahao/Quantization/buddy-mlir/examples/Quantization/input_data.txt";
    std::string paramsFilePath = "/home/xujiahao/Quantization/buddy-mlir/examples/Quantization/params_data.txt";

    std::vector<std::vector<float>> inputData = readDataFromFile(inputFilePath, 10, 6);
    std::vector<std::vector<float>> paramsData = readDataFromFile(paramsFilePath, 10, 6);

    std::vector<float> flatInputData;
    for (const auto &row : inputData) {
        flatInputData.insert(flatInputData.end(), row.begin(), row.end());
    }

    std::vector<float> flatParamsData;
    for (const auto &row : paramsData) {
        flatParamsData.insert(flatParamsData.end(), row.begin(), row.end());
    }
    
    std::vector<_Float16> flatResultData(10 * 3 * 2, 0.0f); 

    printVector(flatInputData, "Flat Input Data");
    printVector(flatParamsData, "Flat Params Data");

    intptr_t input_sizes[3] = {10, 3, 2};
    intptr_t params_sizes[3] = {10, 3, 2};   
    intptr_t result_sizes[3] = {10, 3, 2};

    MemRef<float, 3> input(flatInputData.data(), input_sizes);
    MemRef<float, 3> params(flatParamsData.data(), params_sizes);
    MemRef<_Float16, 3> result(flatResultData.data(), result_sizes);
    

    _mlir_ciface_forward(&result, &params, &input);

    _Float16 *resultData = result.getData();
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 2; ++k) {
                std::cout << resultData[i * 3 * 2 + j * 2 + k] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}