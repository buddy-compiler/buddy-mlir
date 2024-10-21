add_mlir_library(LlamaOptimizations
  AddConstReshape2Const.cpp

  LINK_LIBS PUBLIC
  BuddyUtils
  )