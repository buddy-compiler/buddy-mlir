#include "TimeManager.h"

using namespace buddy::runtime;

void TimeManager::processTimingData() {
  int size = events.size();
  std::map<int, int> times;
  for (int i = 0; i < size; i++) {
    int during = events[i].getDuration();
    std::cout << "idx: " << events[i].getLabel() << " during: " << during
              << std::endl;
    times[events[i].getLabel()] = during;
  }

  using json = nlohmann::json;

  // 输出符号表json
  json jsonObject;

  for (size_t i = 0; i < times.size(); ++i) {
    jsonObject[std::to_string(i)] = times[i];
  }

  // 将 JSON 对象保存到文件
  std::ofstream file("/home/gaoshihao/project/buddy-mlir/examples/"
                     "TimingDialect/tims.json");
  if (file.is_open()) {
    // 使用 dump(4) 以格式化的方式输出，4 个空格缩进
    file << jsonObject.dump(4);
    file.close();
    std::cout << "数据已保存到 times.json 文件中。" << std::endl;
  } else {
    std::cerr << "无法打开文件进行写入。" << std::endl;
  }
}