add_mlir_library(MatMulOptimization
	MatMulOptimize.cpp
  MatMulVectorization.cpp
  MatMulParallelVectorization.cpp
  BatchMatMulOptimize.cpp
  BatchMatMulTileOptimize.cpp
  BatchMatMulSCFOptimize.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)

add_mlir_library(BatchMatMulOptimization
  BatchMatMulOptimize.cpp
)

add_mlir_library(BatchMatMulTileOptimization
  BatchMatMulTileOptimize.cpp
)

add_mlir_library(BatchMatMulSCFOptimization
  BatchMatMulSCFOptimize.cpp
)

add_mlir_library(MatMulParallelVectorization
  MatMulParallelVectorization.cpp
)
