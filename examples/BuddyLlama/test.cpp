#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llama_inference.h"

using namespace std;
using namespace mlir;


int main() {

    // 运行llama推理
    int generateLen = MaxTokenLength - inputContainer.getTokenCnt();
    for (int i = 0; i < generateLen; i++) {
        const auto inferenceStart = std::chrono::high_resolution_clock::now();
        // 执行模型的前向传播
        _mlir_ciface_forward(resultContainer, &paramsContainer, &inputContainer);

        const auto inferenceEnd = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;

        // 确定生成的token
        int tokenIndex = inputContainer.getTokenCnt() - 1;
        const float *startPtr =
        resultContainer[1].getData() + tokenIndex * MaxVocabSize;
        const float *endPtr = startPtr + MaxVocabSize;
        int maxIndex = findMaxIndex(startPtr, endPtr);
        string tok = inputContainer.getStr(maxIndex);
        // 打印生成的token和推理时间
        printIterInfo(i, tok, inferenceTime.count() / 1000);

        // 如果生成了分隔符token（2，</s>）或换行符token（13 <0x0A>），则停止
        if (maxIndex == 2) {
            break;
        }
        // 将生成的token添加到输入和输出容器中
        inputContainer.appendTokenIdx(maxIndex);
        outputContainer.appendTokenIdx(maxIndex);
        free(resultContainer[0].release());
        free(resultContainer[1].release());
    }

    // 打印最终结果
    cout << "
        \033[33;1m[Input]\033[0m " << inputStr << endl;
    cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertLlama()
        << endl;

    return 0;
}

