#include <iostream>
#include <mlir/IR/Operation.h>

namespace buddy {
namespace runtime {

class TimeEvent {
public:
  using OpPtr = mlir::Operation *;

  TimeEvent(OpPtr opPtr)
      : opPtr(opPtr), opName(opPtr->getName().getStringRef().str()),
        startTimestamp(0), endTimestamp(0), duration(0) {}
  TimeEvent(const TimeEvent &) = default;
  TimeEvent(TimeEvent &&) = default;

  void setStartTimestamp() {
    // std::cout << "setStartTimestamp" << std::endl;
    startTimestamp = getCurrentTimeStamp();
  };
  double getStartTimestamp() const { return startTimestamp; };

  void setEndTimestamp() {
    // std::cout << "setEndTimestamp" << std::endl;
    endTimestamp = getCurrentTimeStamp();
  };

  void updateDuration() {
    // std::cout << "updateDuration" << std::endl;
    double interval = getInterval();
    double curDuration = getDuration();
    setDuration(curDuration + interval);
  }

  double getEndTimestamp() const { return endTimestamp; };

  double getInterval() const { return endTimestamp - startTimestamp; }

  double getDuration() const { return duration; }

  void setDuration(double d) { duration = d; }

  OpPtr getOpPtr() const { return opPtr; }

  std::string getOpName() const { return opName; }

private:
  double getCurrentTimeStamp();

  OpPtr opPtr;
  std::string opName;
  double startTimestamp;
  double endTimestamp;
  double duration;
};

} // namespace runtime
} // namespace buddy
