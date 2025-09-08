add_mlir_library(LinalgToVIRPass
	LinalgToVIRPass.cpp

	LINK_LIBS PUBLIC
	MLIRArithDialect
	MLIRLinalgDialect
	MLIRMemRefDialect
	MLIRIR
	MLIRAffineUtils
	MLIRTransformUtils
	VIR
)
