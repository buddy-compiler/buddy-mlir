set(LLVM_Vectorize_DIR ${LLVM_MAIN_SRC_DIR}/lib/Transforms/Vectorize)

add_llvm_component_library(LLVMBuddyVectorize
  ${LLVM_Vectorize_DIR}/LoadStoreVectorizer.cpp
  ${LLVM_Vectorize_DIR}/LoopVectorizationLegality.cpp
  ${LLVM_Vectorize_DIR}/LoopVectorize.cpp
  ${LLVM_Vectorize_DIR}/SLPVectorizer.cpp
  ${LLVM_Vectorize_DIR}/Vectorize.cpp
  ${LLVM_Vectorize_DIR}/VectorCombine.cpp
  ${LLVM_Vectorize_DIR}/VPlan.cpp
  ${LLVM_Vectorize_DIR}/VPlanHCFGBuilder.cpp
  ${LLVM_Vectorize_DIR}/VPlanRecipes.cpp
  ${LLVM_Vectorize_DIR}/VPlanSLP.cpp
  ${LLVM_Vectorize_DIR}/VPlanTransforms.cpp
  ${LLVM_Vectorize_DIR}/VPlanVerifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize

  DEPENDS
  buddy_intrinsics_gen

  LINK_COMPONENTS
  BuddyAnalysis
  BuddyCore
  Support
  BuddyTransformUtils
  )
