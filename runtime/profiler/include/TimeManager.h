
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "TimeEvent.h"

#include <json.hpp>

// class Label {
// public:
//   Label(std::string n) : name(n) {}
//   std::string name;
// };

namespace buddy {
namespace runtime {

class TimeManager {
public:
  TimeManager() = default;
  ~TimeManager() = default;
  TimeManager(const TimeManager &) = delete;
  TimeManager &operator=(const TimeManager &) = delete;

  static void timingStart(TimeEvent *e) { (*e).setStartTimestamp(); }

  static void timingEnd(TimeEvent *e) {
    (*e).setEndTimestamp();
    (*e).updateDuration();
  }

  // 添加计时事件
  template <typename T> static void addEvent(T &&event) {
    events.push_back(std::forward<T>(event));
  }

  static TimeEvent *eventsBack() { return &(events.back()); }

  // 处理并输出计时数据
  void processTimingData();

private:
  // 存储所有的计时事件
  static std::vector<TimeEvent> events;
};

} // namespace runtime
} // namespace buddy