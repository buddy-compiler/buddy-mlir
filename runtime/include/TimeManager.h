#include "TimeEvent.h"
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// class Label {
// public:
//   Label(std::string n) : name(n) {}
//   std::string name;
// };

namespace buddy {
namespace runtime {

class TimeManager {
public:
  TimeManager(const TimeManager &) = delete;
  TimeManager &operator=(const TimeManager &) = delete;

  static TimeManager &instance() {
    static TimeManager _;
    return _;
  }

  static void timingStart() {
    int idx = getCounter();
    auto te = events[idx];
    te.setStartTimestamp();
  }

  static void timingEnd() {
    int idx = getCounter();
    auto te = events[idx];
    te.setEndTimestamp();
    counterPlus1();
  }

  // 添加计时事件
  void addEvent(const TimeEvent &event) { events.push_back(event); }

  // 处理并输出计时数据
  void processTimingData();

private:
  // 私有构造函数，防止外部创建实例
  TimeManager() = default;

  static int getCounter() { return counter; }

  static void counterPlus1() { counter++; }

  inline static int counter = 0;

  // 存储所有的计时事件
  static std::vector<TimeEvent> events;
};

} // namespace runtime
} // namespace buddy