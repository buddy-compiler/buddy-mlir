#include "TimeEvent.h"
#include <chrono>

using namespace buddy::runtime;
using namespace std::chrono;

double TimeEvent::getCurrentTimeStamp() {

  auto now = high_resolution_clock::now();
  auto time = time_point_cast<microseconds>(now);
  return time.time_since_epoch().count() / 1'000.0; // 转换为毫秒
}