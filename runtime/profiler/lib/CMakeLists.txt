include_directories(../include)

add_mlir_library(BuddyRuntime
  Interface.cpp
  TimeEvent.cpp
  TimeManager.cpp
  Profiler.cpp
  )