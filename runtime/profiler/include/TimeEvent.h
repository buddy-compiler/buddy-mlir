namespace buddy {
namespace runtime {

using Label = int;

class TimeEvent {
public:
  TimeEvent(const Label label)
      : label(label), startTimestamp(0.0), endTimestamp(0.0) {}

  void setStartTimestamp() { startTimestamp = getCurrentTimeStamp(); };
  double getStartTimestamp() const { return startTimestamp; };

  void setEndTimestamp() { endTimestamp = getCurrentTimeStamp(); };
  double getEndTimestamp() const { return endTimestamp; };

  double getDuration() const { return endTimestamp - startTimestamp; }

  Label getLabel() const { return label; }

private:
  double getCurrentTimeStamp();

  Label label;
  double startTimestamp;
  double endTimestamp;
};

} // namespace runtime
} // namespace buddy
