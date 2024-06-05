if(FeGen)
  add_subdirectory(FrontendGen)
endif()
<<<<<<< HEAD
=======

>>>>>>> 8cb8a2b9eb8a15bf99280049e7f321d3114fb4bc
add_subdirectory(Interfaces)

if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
  add_subdirectory(Python)
endif()
