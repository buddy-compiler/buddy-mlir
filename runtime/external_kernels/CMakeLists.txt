# -----------------------------------------------------------------------------
# External-call kernel libraries grouped by backend/domain.
# -----------------------------------------------------------------------------

add_subdirectory(rng)

if(BUDDY_RUNTIME_MATMUL_TORQ)
  add_subdirectory(matmul)
endif()
