add_mlir_library(MatMulOptimization
  MatMulOptimize.cpp
  MatMulVectorization.cpp
  MatMulVectorizationDecode.cpp
  MatMulParallelVectorization.cpp
  MatMulBlisVectorization.cpp
  BatchMatMulOptimize.cpp
  BatchMatMulTileOptimize.cpp
  BatchMatMulSCFOptimize.cpp
  BatchMatMulVectorizationDecode.cpp
  MatMulTransposeBVec.cpp
  BatchMatMulOptimize.cpp
  BatchMatMulTileOptimize.cpp
  BatchMatMulSCFOptimize.cpp
  BatchMatMulTransBVec.cpp
  MatMulTransposeBUnrollVec.cpp
  MatmulAMX.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)

add_mlir_library(BatchMatMulOptimization
  BatchMatMulOptimize.cpp
)

add_mlir_library(MatMulParallelVectorization
  MatMulParallelVectorization.cpp
)

add_mlir_library(MatMulTransposeBVec
  MatMulTransposeBVec.cpp
)

add_mlir_library(MatMulBlisVectorization
  MatMulBlisVectorization.cpp
)
