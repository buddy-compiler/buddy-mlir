add_mlir_library(RVProfInstrumentationPass
  RVProfInstrumentation.cpp

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  )
