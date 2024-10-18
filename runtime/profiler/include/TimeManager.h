
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
  ~TimeManager() {
    for (TimeEvent *e : events) {
      delete e;
    }
  }
  TimeManager(const TimeManager &) = delete;
  TimeManager &operator=(const TimeManager &) = delete;

  void timingStart(TimeEvent *e) { (*e).setStartTimestamp(); }

  void timingEnd(TimeEvent *e) {
    (*e).setEndTimestamp();
    (*e).updateDuration();
  }

  // 添加计时事件
  void addEvent(TimeEvent *event) { events.push_back(event); }

  TimeEvent *eventsBack() { return events.back(); }

  // 处理并输出计时数据
  void processTimingData();

private:
  // 存储所有的计时事件
  std::vector<TimeEvent *> events;
};

} // namespace runtime
} // namespace buddy