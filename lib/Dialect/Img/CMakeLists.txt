message(${BUDDY_INCLUDE_DIR})

add_mlir_dialect_library(Img
  ImgDialect.cpp
  ImgOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}/include/Dialect

	LINK_LIBS PUBLIC
	MLIRIR
	)
