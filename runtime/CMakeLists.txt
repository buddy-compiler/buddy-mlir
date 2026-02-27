# -----------------------------------------------------------------------------
# Runtime helper libraries (shared) for Python-driven execution.
# -----------------------------------------------------------------------------

add_library(buddy_rng_utils SHARED buddy_rng_utils.cpp)
target_compile_features(buddy_rng_utils PRIVATE cxx_std_17)
set_target_properties(
  buddy_rng_utils
  PROPERTIES
    OUTPUT_NAME "buddy_rng_utils"
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
)
