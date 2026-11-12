add_mlir_library(LinalgToVIRPass
	LinalgToVIRPass.cpp

	LINK_LIBS PUBLIC
	MLIRArithDialect
	MLIRLinalgDialect
	MLIRMemRefDialect
	MLIRMemRefUtils
	MLIRIR
	MLIRAffineUtils
	MLIRTransformUtils
	VIR
)
