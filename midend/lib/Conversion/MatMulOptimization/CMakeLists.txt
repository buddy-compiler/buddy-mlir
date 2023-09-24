add_mlir_library(MatMulOptimization
  BatchMatMulOptimize.cpp
	MatMulOptimize.cpp
  MatMulVectorization.cpp

  LINK_LIBS PUBLIC
  BuddyUtils
)

add_mlir_library(BatchMatMulOptimization
  BatchMatMulOptimize.cpp
)
