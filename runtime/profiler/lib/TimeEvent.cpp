#include "TimeEvent.h"
#include <chrono>

namespace buddy {
namespace runtime {
double TimeEvent::getCurrentTimeStamp() {

  auto now = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::time_point_cast<std::chrono::microseconds>(now);
  return time.time_since_epoch().count() / 1'000.0; // 转换为毫秒
}
} // namespace runtime
} // namespace buddy
