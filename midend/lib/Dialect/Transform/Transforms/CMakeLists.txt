add_mlir_dialect_library(BuddyTransformPasses
  TransformDialectInterpreter.cpp
  TransformInsertion.cpp

  DEPENDS
  BuddyTransformPassIncGen
#   MLIRLinalgExtTransformOps
#   MLIRTransformExtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRPDLDialect
  MLIRTransformDialect
#   MLIRLinalgExtTransformOps
)
