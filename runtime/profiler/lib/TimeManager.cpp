#include "TimeManager.h"

namespace buddy {
namespace runtime {

void TimeManager::processTimingData(const std::string &outputFilepath) {

  if (events.empty()) {
    std::cerr << "Events list is empty." << std::endl;
    return;
  }

  std::map<std::string, double> times;

  double duration = 0;

  int size = events.size();
  std::cout << size << std::endl;
  for (int i = 0; i < size; i++) {
    if (events[i] == nullptr) {
      std::cerr << "Null pointer found in events at index " << i << std::endl;
      continue;
    }
    mlir::Operation *opPtr = (events[i])->getOpPtr();
    if (opPtr == nullptr) {
      std::cerr << "Null pointer found in opPtr at index " << i << std::endl;
      continue;
    }

    std::string opName = events[i]->getOpName();

    std::cout << opName << std::endl;
    duration = (events[i])->getDuration();
    // std::cout << "OP name: " << opName << " duration: " << duration <<
    // std::endl;
    if (times.find(opName) != times.end())
      times[opName] += duration;
    else
      times[opName] = duration;
  }

  // 输出符号表json
  nlohmann::json jsonObject(times);

  // 将 JSON 对象保存到文件
  std::ofstream file(outputFilepath.c_str());
  if (file.is_open()) {
    file << jsonObject.dump(4);
    file.close();
    std::cout << "Results is saved in " + outputFilepath << std::endl;
  } else {
    std::cerr << "Can't open " + outputFilepath << std::endl;
  }
}
} // namespace runtime
} // namespace buddy
