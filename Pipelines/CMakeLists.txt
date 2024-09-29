add_subdirectory(GPU)

add_mlir_library(BuddyPipelines
    LinalgTensorOpt.cpp
    BufferizeOpt.cpp
    LinalgMemrefOpt.cpp
    
    LINK_LIBS PUBLIC
    MLIRIR
    BuddyGPUPipelines
    BuddyTransformPasses
    BuddyGPUPasses
    BuddyGemmCodegenUtils
    BuddyPipelineUtils
)