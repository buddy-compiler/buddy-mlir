set(LLVM_TARGET_DEFINITIONS TorchLlamaOps.td)
mlir_tablegen(TorchLlamaOps.h.inc -gen-op-decls)
mlir_tablegen(TorchLlamaOps.cpp.inc -gen-op-defs)

add_mlir_dialect(TorchLlamaOps torchllama)
add_mlir_doc(TorchLlamaDialect -gen-dialect-doc TorchLlamaDialect TorchLlama/)
add_mlir_doc(TorchLlamaOps -gen-op-doc TorchLlamaOps TorchLlama/)
