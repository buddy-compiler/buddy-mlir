# -----------------------------------------------------------------------------
# RNG helpers for external-call based lowering paths.
# -----------------------------------------------------------------------------

add_library(buddy_external_rng SHARED RNGUtils.cpp)
target_compile_features(buddy_external_rng PRIVATE cxx_std_17)
install(TARGETS buddy_external_rng
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_runtime)
