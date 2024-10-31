add_mlir_library(OpSimplification
  AddConstReshape2Const.cpp

  LINK_LIBS PUBLIC
  BuddyUtils
  )
