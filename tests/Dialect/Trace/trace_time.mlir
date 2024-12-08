// RUN: buddy-opt %s --lower-trace | FileCheck %s 

module{
func.func private @rtclock() -> f64

func.func @main(){
  // CHECK: %0 = call @rtclock() : () -> f64
  %1 = trace.time_start : -> f64 
  // CHECK: %1 = call @rtclock() : () -> f64
  %2 = trace.time_end : -> f64
  return
}
}
