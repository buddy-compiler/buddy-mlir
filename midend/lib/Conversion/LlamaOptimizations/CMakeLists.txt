add_mlir_library(LlamaOptimizations
  AddConstReshape2Const.cpp
  ReciprocalToConst.cpp

  LINK_LIBS PUBLIC
  BuddyUtils
  )