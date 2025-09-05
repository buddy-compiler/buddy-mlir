add_mlir_library(LinearSiluFusion
  LinearSiluLower.cpp
  LinearSiluFusion.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
) 