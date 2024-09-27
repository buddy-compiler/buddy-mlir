
#include "json.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

// class Label {
// public:
//   Label(std::string n) : name(n) {}
//   std::string name;
// };

class TimingEvent {
public:
  TimingEvent(const int label, double timeStamp, bool isStart)
      : label(label), timeStamp(timeStamp), isStart(isStart) {}

  const int label;
  const double timeStamp;
  const bool isStart; // true 表示 start，false 表示 end
};

class TimingManager {
public:
  TimingManager(const TimingManager &) = delete;
  TimingManager &operator=(const TimingManager &) = delete;

  static TimingManager &instance() {
    static TimingManager _;
    return _;
  }

  static int getCounter() { return counter; }

  static void counterPlus1() { counter++; }

  // 添加计时事件
  void addEvent(const TimingEvent &event) { events.push_back(event); }

  // 处理并输出计时数据
  void processTimingData() {
    int size = events.size();
    std::map<int, int> times;
    for (int i = 0; i + 1 < size; i += 2) {
      int during = events[i + 1].timeStamp - events[i].timeStamp;
      std::cout << "idx: " << events[i].label << " during: " << during
                << std::endl;
      times[events[i].label] = during;
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

private:
  // 私有构造函数，防止外部创建实例
  TimingManager() = default;

  inline static int counter = 0;

  // 存储所有的计时事件
  std::vector<TimingEvent> events;
};

// 获取当前时间戳，单位为毫秒（ms）
double getCurrentTimeStamp() {
  using namespace std::chrono;
  auto now = high_resolution_clock::now();
  auto time = time_point_cast<microseconds>(now);
  return time.time_since_epoch().count() / 1'000.0; // 转换为毫秒
}

// Start
void timingStart() {
  double timeStamp = getCurrentTimeStamp();
  TimingEvent event(TimingManager::getCounter(), timeStamp, true);
  TimingManager::instance().addEvent(event);
  TimingManager::counterPlus1();
}

// End
void timingEnd() {
  double timeStamp = getCurrentTimeStamp();
  TimingEvent event(TimingManager::getCounter(), timeStamp, false);
  TimingManager::instance().addEvent(event);
}

extern "C" void _mlir_ciface_timingStart() {
  timingStart();
  // std::cout << "timing start" << std::endl;
}

extern "C" void _mlir_ciface_timingEnd() {
  timingEnd();
  // std::cout << "timing end" << std::endl;
}
