add_mlir_library(MatMulOptimization
	MatMulOptimize.cpp
  MatMulVectorization.cpp
  MatMulParallelVectorization.cpp
  BatchMatMulTileOptimize.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)

add_mlir_library(BatchMatMulOptimization
  BatchMatMulTileOptimize.cpp
)

add_mlir_library(MatMulParallelVectorization
  MatMulParallelVectorization.cpp
)
