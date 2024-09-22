
#include <chrono>
#include <iostream>
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
  // 禁用拷贝构造和赋值
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
    std::cout << events[1].timeStamp - events[0].timeStamp << std::endl;
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

extern "C" void _mlir_ciface_timingStart() { timingStart(); }

extern "C" void _mlir_ciface_timingEnd() { timingEnd(); }