add_mlir_library(GEMMOptimization
  LoopOrderChange.cpp
  DataCopyGenerate.cpp
  AffineSuperTilling.cpp
  LoopPerfect.cpp
  Eliminate.cpp
  GEMMOptimize.cpp
  )
