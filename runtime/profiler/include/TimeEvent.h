#include "mlir/IR/Operation.h"

namespace buddy {
namespace runtime {

class TimeEvent {
public:
  using OpPtr = mlir::Operation *;

  TimeEvent(const OpPtr opPtr)
      : opPtr(opPtr), startTimestamp(0.0), endTimestamp(0.0), duration(0.0) {}
  TimeEvent(const TimeEvent &) {}
  TimeEvent(TimeEvent &&) {}

  void setStartTimestamp() { startTimestamp = getCurrentTimeStamp(); };
  double getStartTimestamp() const { return startTimestamp; };

  void setEndTimestamp() { endTimestamp = getCurrentTimeStamp(); };

  void updateDuration() {
    double interval = getInterval();
    double curDuration = getDuration();
    setDuration(curDuration + interval);
  }

  double getEndTimestamp() const { return endTimestamp; };

  double getInterval() const { return endTimestamp - startTimestamp; }

  double getDuration() const { return duration; }

  void setDuration(double d) { duration = d; }

  OpPtr getOpPtr() const { return opPtr; }

private:
  double getCurrentTimeStamp();

  OpPtr opPtr;
  double startTimestamp;
  double endTimestamp;
  double duration;
};

} // namespace runtime
} // namespace buddy
