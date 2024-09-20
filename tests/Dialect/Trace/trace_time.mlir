// RUN: buddy-opt %s --lower-trace | FileCheck %s -check-prefix=CHECK

module{
func.func private @rtclock() -> f64

func.func @main(){
  // CHECK: call @rtclock() : () -> f64
  %1 = trace.time_start : -> f64 
  // CHECK: call @rtclock() : () -> f64
  %2 = trace.time_end : -> f64
  return
}
}
