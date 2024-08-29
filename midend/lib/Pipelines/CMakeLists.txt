add_subdirectory(GPU)

add_mlir_library(BuddyPipelines
    LinalgTensorOpt.cpp
    GPU/Utils.cpp
    LINK_LIBS PUBLIC
    MLIRIR
    BuddyGPUPipelines
    BuddyTransformPasses
)