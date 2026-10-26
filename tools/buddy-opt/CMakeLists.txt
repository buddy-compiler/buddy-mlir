get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_tool(buddy-opt
  buddy-opt.cpp
)

target_link_libraries(buddy-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRToLLVMIRTranslationRegistration
  MLIROptLib
  CBConvVectorization
  BuddyBud
  LowerBudPass
  BuddyDIP
  LowerDIPPass
  BuddyDAP
  LowerDAPPass
  BuddyTrace
  ExtendDAPPass
  DAPVectorization
  BuddyRVV
  LowerRVVPass
  MatMulOptimization
  BatchMatMulOptimization
  MatMulParallelVectorization
  TransposeOptimization
  GraphRedundancyElimination
  ConvOptimization
  DepthwiseConvOptimization
  VectorExp
  LowerVectorExpPass
  BuddyGemmini
  LowerGemminiPass
  LowerLinalgToBOSCAMEPass
  LowerLinalgToGemminiPass
  LowerLinalgToIMEPass
  BuddyIME
  BuddyIMETransforms
  LowerIMEPass
  BuddyAME
  BuddyAMETransforms
  BuddyXTAME
  BuddyXTAMETransforms
  BuddyBOSCAME
  BuddyBOSCAMETransforms
  MLIRGPUPasses
  MLIRTransformDialect
  MLIRTransformDialectUtils
  BuddyGPUTransformOPs
  MLIRTestTransforms
  MLIRTestTransformDialect
  MLIRTransforms
  MLIRTransformUtils
  MatMulTransposeBVec
  LinalgToVIRPass
  VIRToVectorPass
  TosaElementwiseFusion
  AssumeTightMemRefLayoutPass
  StaticizeMemRefLayoutPass
  EliminateMemRefCopyPass
  EliminateLargeZeroConstants
  TraceInsert
  )

# Dialect and pass plugins resolve MLIR symbols against buddy-opt.
export_executable_symbols_for_plugins(buddy-opt)
